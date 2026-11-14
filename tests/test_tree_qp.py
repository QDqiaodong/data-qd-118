"""Independent checks for the exact tree QP.

The small-tree oracle here is deliberately naive: it enumerates *every* subset
of leaves to clamp to zero, keeps the feasible subsets, and evaluates the true
objective with an independent dense linear solve.  That is exponential and is
used only as a cross-check on tiny trees; the production solver must match it
exactly.  The wide 200-node test then shows that the same solver finishes
without ever doing that enumeration.
"""
import random
import unittest
from fractions import Fraction as F

from demand_backtest.tree_qp import objective, reconcile


def structure(series, parents):
    index = {sid: i for i, sid in enumerate(series)}
    children = [[] for _ in series]
    for i, parent in enumerate(parents):
        if parent is not None:
            children[index[parent]].append(i)
    root = parents.index(None)
    leaves = [i for i, kids in enumerate(children) if not kids]
    return children, root, leaves


def dense_solution(series, parents, weights, forecasts, clamped):
    """Independent equality-constrained optimum with `clamped` leaves at zero.

    Solves the reduced dense normal equations over the free leaf values:
    g_i = sum_{a in subtree_i} w_a * (sum_{l in subtree_a} x_l - f_a)^2.
    This is a different formulation from the elimination recurrence in
    tree_qp, so agreement is a genuine cross-check.
    """
    children, root, leaves = structure(series, parents)
    subtrees = {}
    for i in range(len(series)):
        members, stack = [], [i]
        while stack:
            node = stack.pop()
            if children[node]:
                stack.extend(children[node])
            else:
                members.append(node)
        subtrees[i] = members
    free = [leaf for leaf in leaves if leaf not in clamped]
    if not free:
        return [F(0)] * len(series)
    position = {leaf: k for k, leaf in enumerate(free)}
    size = len(free)
    matrix = [[F(0)] * (size + 1) for _ in range(size)]
    rhs = [F(0)] * size
    for i, weight in enumerate(weights):
        members = [leaf for leaf in subtrees[i] if leaf in position]
        for first in members:
            rhs[position[first]] += 2 * weight * forecasts[i]
            for second in members:
                matrix[position[first]][position[second]] += 2 * weight
    for row in range(size):
        matrix[row][size] = rhs[row]
    for column in range(size):
        pivot = next((row for row in range(column, size)
                      if matrix[row][column] != 0), None)
        if pivot is None:
            raise ValueError('singular reference system')
        matrix[column], matrix[pivot] = matrix[pivot], matrix[column]
        scale = matrix[column][column]
        matrix[column] = [value / scale for value in matrix[column]]
        for row in range(size):
            if row != column and matrix[row][column] != 0:
                factor = matrix[row][column]
                matrix[row] = [a - factor * b
                               for a, b in zip(matrix[row], matrix[column])]
    leaf_value = {leaf: matrix[position[leaf]][size] for leaf in free}
    values = [F(0)] * len(series)
    for i in range(len(series)):
        values[i] = sum((leaf_value[leaf] for leaf in subtrees[i]
                         if leaf in leaf_value), F(0))
    return values


def enumerate_optimum(series, parents, weights, forecasts):
    """Exponential oracle: try every clamped-leaf subset, keep feasible ones."""
    children, root, leaves = structure(series, parents)
    best = None
    for mask in range(1 << len(leaves)):
        clamped = {leaves[k] for k in range(len(leaves)) if mask >> k & 1}
        values = dense_solution(series, parents, weights, forecasts, clamped)
        if any(values[leaf] < 0 for leaf in leaves if leaf not in clamped):
            continue
        gradients = leaf_gradients(series, parents, weights, forecasts, values)
        if any(gradients[leaf] != 0 for leaf in leaves if leaf not in clamped):
            continue
        score = objective(weights, forecasts, values)
        if best is None or score < best:
            best = score
    return best


def leaf_gradients(series, parents, weights, forecasts, values):
    children, root, leaves = structure(series, parents)
    slopes = [2 * weights[i] * (values[i] - forecasts[i])
              for i in range(len(series))]
    index = {sid: i for i, sid in enumerate(series)}
    parent_of = {i: index[parents[i]] for i in range(len(series))
                 if parents[i] is not None}
    gradients = {}
    for leaf in leaves:
        # Walk every ancestor explicitly; this avoids reusing solver internals.
        total, node = F(0), leaf
        while True:
            total += slopes[node]
            if node not in parent_of:
                break
            node = parent_of[node]
        gradients[leaf] = total
    return gradients


def solve(series, parents, weights, forecasts):
    children, root, leaves = structure(series, parents)
    return reconcile(children, root, weights, forecasts, leaves)


def random_tree(rng, size):
    series = [f's{i}' for i in range(size)]
    parents = [None] + [f's{rng.randrange(0, i)}' for i in range(1, size)]
    return series, parents


class SolverOracleTests(unittest.TestCase):
    def test_small_trees_match_exponential_enumeration(self):
        rng = random.Random(20261114)
        checked = 0
        for _ in range(2500):
            size = rng.randrange(1, 10)
            series, parents = random_tree(rng, size)
            children, _, leaves = structure(series, parents)
            if len(leaves) > 9:
                continue
            weights = [F(rng.choice([1, 1, 2, 3, 5]), rng.choice([1, 1, 2, 3]))
                       for _ in range(size)]
            forecasts = [F(rng.randrange(-6, 10), rng.choice([1, 1, 2, 3]))
                         for _ in range(size)]
            values, gradients, clamped, _ = solve(series, parents, weights,
                                                  forecasts)
            self.assertEqual(objective(weights, forecasts, values),
                             enumerate_optimum(series, parents, weights,
                                               forecasts))
            checked += 1
        self.assertGreater(checked, 1500)

    def test_optimum_hits_zero_leaf_exactly(self):
        # One leaf wants a negative value; the optimum pins it at zero and
        # rebalances the sibling rather than clipping and keeping the old sum.
        series = ['all', 'a', 'b']
        parents = [None, 'all', 'all']
        weights = [F(1), F(1), F(1)]
        forecasts = [F(0), F(-10), F(10)]
        values, gradients, clamped, _ = solve(series, parents, weights,
                                              forecasts)
        self.assertEqual(values[1], 0)   # a is pinned at exactly zero
        self.assertEqual(values[2], 5)   # b absorbs the rebalanced total
        self.assertEqual(values[0], 5)
        self.assertEqual(clamped, {1})
        self.assertGreaterEqual(gradients[1], 0)
        self.assertEqual(gradients[2], 0)
        self.assertEqual(objective(weights, forecasts, values),
                         enumerate_optimum(series, parents, weights, forecasts))

    def test_multilayer_tree_and_ordering(self):
        series = ['r', 'm', 'leafx', 'leafy', 'n']
        parents = [None, 'r', 'm', 'm', 'r']
        weights = [F(10), F(3), F(1), F(2), F(5)]
        forecasts = [F(4), F(1), F(-2), F(1), F(6)]
        values, gradients, clamped, _ = solve(series, parents, weights,
                                              forecasts)
        children, _, leaves = structure(series, parents)
        for i, kids in enumerate(children):
            if kids:
                self.assertEqual(values[i], sum((values[k] for k in kids),
                                                F(0)))
        for leaf in leaves:
            self.assertGreaterEqual(values[leaf], 0)
            if values[leaf] == 0:
                self.assertGreaterEqual(gradients[leaf], 0)
            else:
                self.assertEqual(gradients[leaf], 0)
        self.assertEqual(objective(weights, forecasts, values),
                         enumerate_optimum(series, parents, weights, forecasts))
        # Shuffling node order must not change the optimum values.
        order = [4, 2, 0, 3, 1]
        shuffled_series = [series[i] for i in order]
        position = {old: new for new, old in enumerate(order)}
        shuffled_parents = [None if parents[i] is None
                            else shuffled_series[position[series.index(
                                parents[i])]] for i in order]
        shuffled_weights = [weights[i] for i in order]
        shuffled_forecasts = [forecasts[i] for i in order]
        other, _, _, _ = solve(shuffled_series, shuffled_parents,
                               shuffled_weights, shuffled_forecasts)
        self.assertEqual({shuffled_series[i]: other[i] for i in range(5)},
                         {series[i]: values[i] for i in range(5)})

    def test_wide_tree_200_nodes_is_not_exponential(self):
        # 200 nodes, 199 leaves: enumerating 2^199 subsets is impossible, so a
        # fast exact answer here proves the implementation is active-set based.
        size = 200
        series = ['root'] + [f'leaf{i}' for i in range(1, size)]
        parents = [None] + ['root'] * (size - 1)
        rng = random.Random(7)
        weights = [F(1000)] + [F(rng.randrange(1, 50)) for _ in range(size - 1)]
        forecasts = [F(-30)] + [F(rng.randrange(-20, 40))
                                for _ in range(size - 1)]
        values, gradients, clamped, iterations = solve(series, parents,
                                                       weights, forecasts)
        self.assertEqual(values[0], sum(values[1:], F(0)))
        for leaf in range(1, size):
            self.assertGreaterEqual(values[leaf], 0)
            if values[leaf] == 0:
                self.assertGreaterEqual(gradients[leaf], 0)
            else:
                self.assertEqual(gradients[leaf], 0)
        # All-forecast clipping would be feasible here but not optimal; verify
        # the objective beats the naive "clamp negatives then recompute" rule.
        clipped = [max(forecast, F(0)) for forecast in forecasts]
        clipped[0] = sum(clipped[1:], F(0))
        self.assertLessEqual(objective(weights, forecasts, values),
                             objective(weights, forecasts, clipped))
        self.assertLess(iterations, 40)

    def test_root_only_tree(self):
        values, gradients, clamped, _ = solve(['only'], [None], [F(1)],
                                              [F(-5)])
        self.assertEqual(values, [F(0)])
        self.assertEqual(clamped, {0})
        self.assertGreaterEqual(gradients[0], F(0))


if __name__ == '__main__':
    unittest.main()
