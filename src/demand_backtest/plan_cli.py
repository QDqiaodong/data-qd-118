"""Command line entry point for minimum-cost replenishment planning.

    demand-plan --db facts.db --config plan.json --output plan_report.json

The fact store is opened strictly read-only, so a planning run can never change
the source snapshot, and the config file is read with the same strict JSON
rules as the other business inputs.  Validation or infeasibility errors exit 2
and never touch an existing output file; on success the report is published by
an atomic replace.
"""
import argparse
import json
import sqlite3
import sys
from pathlib import Path

from .common import save
from .plan import plan

DESCRIPTION = ('Minimum-cost replenishment plan from a forecast fact store '
               'cross-section')


def main(argv=None):
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('--db', type=Path, required=True,
                        help='existing SQLite forecast fact store (read-only)')
    parser.add_argument('--config', type=Path, required=True,
                        help='planner config: origin/method, series, daily '
                             'capacity, initial batches')
    parser.add_argument('--output', type=Path, required=True,
                        help='JSON plan report (atomic replace on success)')
    args = parser.parse_args(argv)
    try:
        paths = [args.db, args.config]
        resolved = [path.resolve() for path in paths]
        if len(set(resolved)) != len(resolved):
            raise ValueError('--db and --config must be distinct files')
        output = args.output.resolve()
        if output in resolved:
            raise ValueError('output must differ from every input')
        report = plan(args.db, args.config)
        save(args.output, report)
    except (ValueError, TypeError, KeyError, OSError, OverflowError,
            RecursionError, sqlite3.Error) as error:
        print(str(error), file=sys.stderr)
        return 2
    print(json.dumps(report, sort_keys=True, ensure_ascii=False,
                     allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
