"""Command line entry point for hierarchy reconciliation.

    demand-reconcile --report CALENDAR.json --report VINTAGE.json \
                     --hierarchy hierarchy.json --output reconciled.json

Both report inputs are existing, already-published backtest reports.  They are
never written to; only ``--output`` is replaced, and only after the whole
computation succeeded.
"""
import argparse
import json
import sys
from pathlib import Path

from .common import read, save
from .reconcile import analyze_reconcile

DESCRIPTION = 'Reconcile calendar and vintage backtest reports onto a hierarchy'
# Reports are already-published backtests: 200000 facts are roughly 55 MB of
# pretty-printed JSON, so the report bound is higher than a raw business input.
# The hierarchy file stays at the strict 8 MB business-input limit.
_REPORT_BYTES = 128_000_000
_HIERARCHY_BYTES = 8_000_000


def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--report', type=Path, action='append', required=True,
                        help='existing backtest report; exactly one calendar and one vintage')
    parser.add_argument('--hierarchy', type=Path, required=True,
                        help='hierarchy nodes: series/parent/weight')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        if len(args.report) != 2:
            raise ValueError('--report needs exactly two reports')
        paths = list(args.report) + [args.hierarchy]
        if len({path.resolve() for path in paths}) != len(paths):
            raise ValueError('inputs must be three distinct files')
        output = args.output.resolve()
        if any(path.resolve() == output for path in paths):
            raise ValueError('input and output must differ')
        loaded = [read(path, _REPORT_BYTES) for path in args.report]
        hierarchy, hierarchy_digest = read(args.hierarchy, _HIERARCHY_BYTES)
        report = {
            'schema_version': 1, 'mode': 'hierarchy_reconcile',
            'result': analyze_reconcile(
                [data for data, _ in loaded], hierarchy,
                report_sha256=[digest for _, digest in loaded],
                hierarchy_sha256=hierarchy_digest)}
        save(args.output, report)
    except (ValueError, TypeError, KeyError, OSError, OverflowError,
            RecursionError) as error:
        print(str(error), file=sys.stderr)
        return 2
    print(json.dumps(report, sort_keys=True, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
