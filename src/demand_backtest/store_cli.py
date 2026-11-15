"""Command line entry point for the SQLite forecast fact store.

    demand-store apply --db facts.db --manifest batch.json
    demand-store query --db facts.db

``apply`` reads only the reports named in the manifest's ``puts`` entries and
never reruns the backtest model.  Both commands print their JSON result to
stdout; ``query`` never writes.  Validation and conflict failures exit 2 and
leave the logical store unchanged.  The store, the manifest and every source
report must be distinct files, and no source report is ever overwritten.
"""
import argparse
import json
import sqlite3
import sys
from pathlib import Path

from .store import apply_facts, query_facts

DESCRIPTION = 'Apply and query forecast facts in a SQLite fact store'


def main(argv=None):
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    commands = parser.add_subparsers(dest='command', required=True)
    apply_parser = commands.add_parser('apply', help='apply one atomic batch')
    apply_parser.add_argument('--db', type=Path, required=True,
                              help='SQLite fact store (created on first apply)')
    apply_parser.add_argument('--manifest', type=Path, required=True,
                              help='batch manifest: txid, expected_generation,'
                                   ' puts, drops')
    query_parser = commands.add_parser('query', help='read one snapshot')
    query_parser.add_argument('--db', type=Path, required=True,
                              help='existing SQLite fact store')
    args = parser.parse_args(argv)
    try:
        if args.command == 'apply':
            result = apply_facts(args.db, args.manifest)
        else:
            result = query_facts(args.db)
    except (ValueError, TypeError, KeyError, OSError, OverflowError,
            RecursionError, sqlite3.Error) as error:
        print(str(error), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, ensure_ascii=False,
                     allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
