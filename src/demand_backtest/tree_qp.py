"""Exact tree-structured reconciliation QP.

For one forecast group the problem is

    minimize  sum_i weight_i * (x_i - forecast_i)^2
    subject to x_parent = sum of its direct children,   x_leaf >= 0.

Every node is its own constraint, so the equalities are eliminated by a
bottom-up/top-down recurrence instead of a dense linear solve.  The only
inequality is a nonnegativity bound on each leaf, which is handled by a primal
active-set method: leaves currently fixed at zero stay in the working set,
nonnegative leaves must have a zero reduced gradient, and zero leaves must have
a nonnegative reduced gradient (the KKT certificate reported in the output).

The implementation is exact (fractions.Fraction) and never enumerates the
2^(number of leaves) active sets.  Each active-set step either adds a blocking
leaf or releases a leaf with a negative multiplier; the working set changes
monotonically between subproblem solves, so the number of steps is small and
independent of the exponential search space.  A generous hard cap turns any
unexpected degeneracy into an explicit error instead of an unbounded run.
"""

from fractions import Fraction as F

_ZERO = F(0)


def reconcile(children, root, weights, forecasts, leaves):
    """Solve one group exactly.

    ``children[i]`` lists child node indices, ``weights[i]`` and
    ``forecasts[i]`` are Fractions, and ``leaves`` are the leaf indices.  The
    tree must have ``len(weights)`` nodes and exactly one root.
    Returns ``(values, gradients, zero_leaves, iterations)`` where
    ``gradients[l]`` is the exact derivative of the objective with respect to
    leaf ``l`` at the optimum.
    """
    count = len(weights)
    order = []  # parents before children
    stack = [root]
    while stack:
        node = stack.pop()
        order.append(node)
        stack.extend(children[node])
    if len(order) != count:
        raise ValueError('hierarchy must be one connected tree')
    reverse = order[::-1]

    # For a subtree, the optimal value and its sensitivity to the multiplier
    # passed in by the parent are linear: value = a + b * multiplier.  The same
    # holds for the multiplier this subtree passes down: c + d * multiplier.
    a = [_ZERO] * count
    b = [_ZERO] * count
    c = [_ZERO] * count
    d = [_ZERO] * count

    clamped = set()  # leaves fixed at zero (working set)
    values = [_ZERO] * count
    max_steps = 50 * count + 200
    for iterations in range(1, max_steps + 1):
        # Eliminate the equalities bottom-up for the current working set.
        for node in reverse:
            weight = weights[node]
            forecast = forecasts[node]
            kids = children[node]
            if not kids:
                if node in clamped:
                    a[node] = _ZERO
                    b[node] = _ZERO
                else:
                    a[node] = forecast
                    b[node] = F(1, 2) / weight
                continue
            total_a = _ZERO
            total_b = _ZERO
            for kid in kids:
                total_a += a[kid]
                total_b += b[kid]
            denominator = 1 + 2 * weight * total_b
            c[node] = 2 * weight * (forecast - total_a) / denominator
            d[node] = F(1) / denominator
            a[node] = total_a + total_b * c[node]
            b[node] = total_b * d[node]

        # Reconstruct the candidate optimum from the root multiplier (zero).
        candidate = [_ZERO] * count
        stack = [(root, _ZERO)]
        while stack:
            node, multiplier = stack.pop()
            candidate[node] = a[node] + b[node] * multiplier
            if children[node]:
                child_multiplier = c[node] + d[node] * multiplier
                for kid in children[node]:
                    stack.append((kid, child_multiplier))

        # Move from the current feasible point to the candidate.  A leaf
        # crossing zero blocks the step and joins the working set (smallest
        # index first on ties, so degenerate steps still make progress).
        step = F(1)
        blockers = []
        for leaf in leaves:
            if leaf in clamped or candidate[leaf] >= 0:
                continue
            fraction = values[leaf] / (values[leaf] - candidate[leaf])
            if fraction < step:
                step = fraction
                blockers = [leaf]
            elif fraction == step:
                blockers.append(leaf)
        if blockers:
            for node in range(count):
                values[node] += step * (candidate[node] - values[node])
            for leaf in blockers:
                clamped.add(leaf)
                values[leaf] = _ZERO
            continue

        values = candidate

        # Reduced gradient of every leaf: own loss slope plus every ancestor.
        slopes = [2 * weights[i] * (values[i] - forecasts[i])
                  for i in range(count)]
        gradients = [None] * count
        gradients[root] = slopes[root]
        stack = [root]
        while stack:
            node = stack.pop()
            for kid in children[node]:
                gradients[kid] = gradients[node] + slopes[kid]
                stack.append(kid)

        released = [leaf for leaf in leaves
                    if leaf in clamped and gradients[leaf] < 0]
        if not released:
            return values, gradients, clamped, iterations
        clamped.discard(min(released, key=lambda leaf: (gradients[leaf], leaf)))

    raise ValueError('tree reconciliation did not converge')


def objective(weights, forecasts, values):
    """Exact objective value at ``values``."""
    total = _ZERO
    for weight, forecast, value in zip(weights, forecasts, values):
        total += weight * (value - forecast) ** 2
    return total
