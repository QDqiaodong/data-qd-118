"""SQLite forecast fact store with generation receipts.

The store consumes already-published calendar and revision-vintage backtest
reports and never reruns the forecast model.  A fact is keyed by
``(series, origin, target_date, method)`` and keeps the exact ``horizon``,
``forecast``, ``actual`` and ``scale`` values as canonical Fraction text.
Reports that agree on all four values share one fact, and every report name
stays attached to it as provenance.

``apply`` runs as one SQLite transaction.  Reports named in ``drops`` release
their ownership first, conflicts are judged against that post-drop state, and a
fact is deleted only after its last owner disappears.  The transaction runs
only when the caller's ``expected_generation`` equals the committed generation
and then atomically writes facts, ownership, the report-byte catalog, the
ledger entry (content digest plus receipt) and the incremented generation.

A repeated transaction id with identical change content returns the original
receipt, even when its expected generation is now stale; the same transaction
id with different content is rejected.  The content fingerprint binds report
bytes (SHA256), names and drops -- never paths.  ``query`` reads generation,
the report catalog, sorted facts and their owners from a single read snapshot.
"""
import hashlib
import json
import os
import sqlite3
from fractions import Fraction
from pathlib import Path

from .common import ident, number, read, require, rows
from .reconcile import _MAX_FACTS, _fraction, _parse_day, _report, _text

_MAX_REPORT_BYTES = 128_000_000
_MANIFEST_BYTES = 8_000_000
_MAX_PATH = 4096
_MAX_NAMES = 1000
_MAX_GENERATION = 2 ** 63 - 2
_BUSY_SECONDS = 30.0
_BATCH_ROWS = 20000
_FACT_REQUIRED = {'series', 'origin', 'target_date', 'method', 'horizon',
                  'forecast', 'actual', 'scale'}
_FACT_EXTRA = {'actual_published_on', 'error_forecast_minus_actual'}

_SCHEMA = (
    'CREATE TABLE IF NOT EXISTS meta ('
    ' key TEXT PRIMARY KEY, value TEXT NOT NULL) WITHOUT ROWID',
    'CREATE TABLE IF NOT EXISTS reports ('
    ' name TEXT PRIMARY KEY,'
    " kind TEXT NOT NULL CHECK (kind IN ('calendar', 'vintage')),"
    ' source_sha256 TEXT NOT NULL,'
    ' report_sha256 TEXT NOT NULL,'
    ' fact_count INTEGER NOT NULL) WITHOUT ROWID',
    'CREATE TABLE IF NOT EXISTS facts ('
    ' series TEXT NOT NULL, origin TEXT NOT NULL,'
    ' target_date TEXT NOT NULL, method TEXT NOT NULL,'
    ' horizon INTEGER NOT NULL, forecast TEXT NOT NULL,'
    ' actual TEXT NOT NULL, scale TEXT NOT NULL,'
    ' PRIMARY KEY (series, origin, target_date, method)) WITHOUT ROWID',
    'CREATE TABLE IF NOT EXISTS fact_reports ('
    ' series TEXT NOT NULL, origin TEXT NOT NULL,'
    ' target_date TEXT NOT NULL, method TEXT NOT NULL,'
    ' name TEXT NOT NULL REFERENCES reports(name) ON DELETE CASCADE,'
    ' PRIMARY KEY (series, origin, target_date, method, name)) WITHOUT ROWID',
    'CREATE TABLE IF NOT EXISTS ledger ('
    ' txid TEXT PRIMARY KEY, generation INTEGER NOT NULL UNIQUE,'
    ' content_sha256 TEXT NOT NULL, receipt TEXT NOT NULL,'
    ' receipt_sha256 TEXT NOT NULL) WITHOUT ROWID',
    'CREATE INDEX IF NOT EXISTS fact_reports_by_name'
    ' ON fact_reports(name)',
)
_KEY_JOIN = ('r.series=f.series AND r.origin=f.origin AND'
             ' r.target_date=f.target_date AND r.method=f.method')


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode('utf-8')


def _digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _path(value):
    if not isinstance(value, str) or not value or len(value) > _MAX_PATH or \
            '\x00' in value:
        raise ValueError('put path must be a nonempty path string')
    return value


def _nonnegative(value, field):
    parsed = _fraction(value, field)
    if parsed < 0:
        raise ValueError(field + ' must be nonnegative')
    return parsed


def _fact(entry):
    """Validate one report fact and return its normalized key and values."""
    if not isinstance(entry, dict):
        raise ValueError('forecast facts must be objects')
    keys = set(entry)
    if not _FACT_REQUIRED <= keys or not keys <= _FACT_REQUIRED | _FACT_EXTRA:
        raise ValueError('forecast fact has undefined or missing fields')
    sid = ident(entry['series'])
    origin = entry['origin']
    _parse_day(origin, 'origin')
    target = entry['target_date']
    _parse_day(target, 'target_date')
    method = ident(entry['method'])
    step = number(entry['horizon'], 1, 100)
    forecast = _fraction(entry['forecast'], 'forecast')
    raw_actual = entry['actual']
    if type(raw_actual) is int:
        # Backtest reports serialize `actual` as the nonnegative integer
        # observed quantity; keep it exact without imposing a new ceiling.
        if raw_actual < 0:
            raise ValueError('actual must be nonnegative')
        actual = Fraction(raw_actual)
    else:
        actual = _nonnegative(raw_actual, 'actual')
    scale = _nonnegative(entry['scale'], 'scale')
    return (sid, origin, target, method), (step, forecast, actual, scale)


def load_manifest(path):
    """Read and shape-check an apply manifest; report bytes are read later."""
    data, _ = read(path, _MANIFEST_BYTES)
    require(data, 'txid expected_generation puts drops')
    txid = ident(data['txid'])
    expected = number(data['expected_generation'], 0, _MAX_GENERATION)
    seen = set()
    puts = []
    for entry in rows(data['puts'], _MAX_NAMES):
        require(entry, 'name path')
        name = ident(entry['name'])
        location = _path(entry['path'])
        if name in seen:
            raise ValueError('a name appears more than once in one batch')
        seen.add(name)
        puts.append((name, location))
    drops = []
    for entry in rows(data['drops'], _MAX_NAMES):
        require(entry, 'name')
        name = ident(entry['name'])
        if name in seen:
            raise ValueError('a name appears more than once in one batch')
        seen.add(name)
        drops.append(name)
    return {'txid': txid, 'expected_generation': expected, 'puts': puts,
            'drops': drops}


def _prepare(manifest):
    """Read every put report and merge facts, rejecting any disagreement.

    Reports in one batch may share a key only when horizon, forecast, actual
    and scale all agree exactly; otherwise the whole batch is rejected before
    the database is touched.  The content digest covers report bytes and names,
    never their paths.
    """
    catalog, incoming, merged = [], [], {}
    for name, location in manifest['puts']:
        data, report_digest = read(location, _MAX_REPORT_BYTES)
        try:
            kind, source_digest, raw_facts = _report(data)
        except ValueError as error:
            raise ValueError(f'report {name}: {error}') from error
        if len(raw_facts) > _MAX_FACTS:
            raise ValueError('a report exceeds 200000 forecast facts')
        local = {}
        for entry in raw_facts:
            key, values = _fact(entry)
            if key in local:
                raise ValueError(f'report {name} repeats forecast fact '
                                 + '/'.join(key))
            local[key] = values
        catalog.append({'name': name, 'kind': kind,
                        'source_sha256': source_digest,
                        'report_sha256': report_digest,
                        'fact_count': len(raw_facts)})
        incoming.append((name, local))
        for key, values in local.items():
            held = merged.get(key)
            if held is None:
                merged[key] = [values, {name}]
            elif held[0] != values:
                raise ValueError('reports in one batch disagree for '
                                 'forecast fact ' + '/'.join(key))
            else:
                held[1].add(name)
    catalog.sort(key=lambda entry: entry['name'])
    content = {'drops': sorted(manifest['drops']), 'puts': catalog}
    return {'catalog': catalog, 'incoming': incoming, 'merged': merged,
            'content_sha256': _digest(content)}


def _check_paths(db_path, manifest_path, manifest):
    store = Path(db_path).resolve()
    if store == Path(manifest_path).resolve():
        raise ValueError('store and manifest must differ')
    for _, location in manifest['puts']:
        if Path(location).resolve() == store:
            raise ValueError('store must not overwrite a source report')


def _precheck(db_path, manifest, prepared):
    """Reject batches that cannot apply before any database file is created."""
    snapshot = _read_snapshot(db_path, allow_missing=True)
    try:
        if snapshot is None:
            return None
        if 'meta' not in _tables(snapshot):
            raise ValueError('not a forecast fact store')
        schema = snapshot.execute(
            "SELECT value FROM meta WHERE key='schema_version'").fetchone()
        if schema is None or schema[0] != '1':
            raise ValueError('unsupported store schema_version')
        # A replayed transaction is answered from its original receipt before
        # any state checks: the names it dropped or bound already moved on.
        prior = snapshot.execute(
            'SELECT content_sha256, receipt FROM ledger WHERE txid=?',
            (manifest['txid'],)).fetchone()
        if prior is not None:
            if prior['content_sha256'] != prepared['content_sha256']:
                raise ValueError('txid already applied with different content')
            return json.loads(prior['receipt'])
        names = {row[0] for row in snapshot.execute('SELECT name FROM reports')}
        missing = sorted(set(manifest['drops']) - names)
        if missing:
            raise ValueError('cannot drop unknown report: ' + ','.join(missing))
        taken = sorted({entry['name'] for entry in prepared['catalog']} & names)
        if taken:
            raise ValueError('report name already bound: ' + ','.join(taken))
        return None
    finally:
        if snapshot is not None:
            snapshot.close()


def _read_snapshot(db_path, allow_missing=False):
    """Open a query-only snapshot; a missing or empty store is generation 0.

    The connection is opened read-write but pinned with ``query_only`` so
    SQLite can still roll back a hot journal left by a crash before serving the
    snapshot.  The snapshot itself never writes through the query flag.
    """
    path = Path(db_path)
    if path.is_symlink():
        raise ValueError('store cannot be a symlink')
    if not _exists(db_path):
        if allow_missing:
            return None
        raise ValueError('store does not exist')
    if not path.is_file():
        raise ValueError('store path is not a regular file')
    connection = sqlite3.connect(str(path), timeout=_BUSY_SECONDS,
                                 isolation_level=None)
    connection.row_factory = sqlite3.Row
    connection.execute('PRAGMA foreign_keys=ON')
    connection.execute('PRAGMA query_only=ON')
    return connection


def _open_writer(db_path):
    path = Path(db_path)
    if path.is_symlink():
        raise ValueError('store cannot be a symlink')
    if path.exists() and not path.is_file():
        raise ValueError('store path is not a regular file')
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(path), timeout=_BUSY_SECONDS,
                                 isolation_level=None)
    connection.row_factory = sqlite3.Row
    connection.execute('PRAGMA foreign_keys=ON')
    # WAL keeps a reader's snapshot stable without blocking the single writer,
    # so an interleaved query never observes a half-applied batch.
    connection.execute('PRAGMA journal_mode=WAL')
    connection.execute('PRAGMA synchronous=FULL')
    return connection


def _tables(connection):
    return {row[0] for row in connection.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}


def _initialize(connection):
    existing = _tables(connection)
    if 'meta' not in existing:
        if existing:
            raise ValueError('not a forecast fact store')
        for statement in _SCHEMA:
            connection.execute(statement)
        connection.execute(
            "INSERT INTO meta (key, value) VALUES ('schema_version', '1')")
        connection.execute(
            "INSERT INTO meta (key, value) VALUES ('generation', '0')")
        return
    row = connection.execute(
        "SELECT value FROM meta WHERE key='schema_version'").fetchone()
    if row is None or row[0] != '1':
        raise ValueError('unsupported store schema_version')
    for statement in _SCHEMA:
        connection.execute(statement)


def _generation(connection):
    row = connection.execute(
        "SELECT value FROM meta WHERE key='generation'").fetchone()
    if row is None:
        raise ValueError('store metadata is incomplete')
    # The committed generation may sit one above the largest accepted
    # expected_generation, since each successful apply increments it.
    return number(int(row[0]), 0, _MAX_GENERATION + 1)


def _catalog(connection):
    return [{'name': row['name'], 'kind': row['kind'],
             'source_sha256': row['source_sha256'],
             'report_sha256': row['report_sha256'],
             'fact_count': row['fact_count']}
            for row in connection.execute('SELECT * FROM reports'
                                          ' ORDER BY name')]


def _receipt(manifest, content_sha256, generation, counts, catalog):
    receipt = {'txid': manifest['txid'], 'generation': generation,
               'content_sha256': content_sha256,
               'facts_added': counts['added'],
               'facts_removed': counts['removed'],
               'facts_shared': counts['shared'],
               'fact_count': counts['total'],
               'reports': catalog, 'dropped': sorted(manifest['drops'])}
    text = json.dumps(receipt, sort_keys=True, ensure_ascii=False,
                      allow_nan=False, separators=(',', ':'))
    return receipt, text, hashlib.sha256(text.encode('utf-8')).hexdigest()


def apply_facts(db_path, manifest_path):
    """Apply one batch atomically and return its (possibly replayed) receipt.

    Report parsing, within-batch merging, path checks, and any replay lookup
    happen before a write lock is taken.  The transaction then re-checks the
    ledger and generation, releases dropped ownership, checks conflicts against
    the post-drop state, and writes facts, ownership, catalog, ledger and
    generation together.  Any failure rolls the whole transaction back, leaving
    the logical store unchanged.
    """
    manifest = load_manifest(manifest_path)
    _check_paths(db_path, manifest_path, manifest)
    prepared = _prepare(manifest)
    exists = _exists(db_path)
    if exists:
        replay = _precheck(db_path, manifest, prepared)
        if replay is not None:
            return replay
    else:
        # Fail before creating a database file when a fresh store cannot accept
        # the batch; the post-transaction checks would only repeat this.
        _fresh_check(manifest)

    connection = _open_writer(db_path)
    try:
        connection.execute('BEGIN IMMEDIATE')
        _initialize(connection)
        action, replay = _state_checks(connection, manifest, prepared)
        if action == 'replay':
            connection.execute('ROLLBACK')
            return replay
        generation = _generation(connection)
        counts = _mutate(connection, manifest, prepared)
        catalog = _catalog(connection)
        receipt, text, receipt_digest = _receipt(
            manifest, prepared['content_sha256'], generation + 1, counts,
            catalog)
        connection.execute(
            "UPDATE meta SET value=? WHERE key='generation'",
            (str(generation + 1),))
        connection.execute(
            'INSERT INTO ledger VALUES (?, ?, ?, ?, ?)',
            (manifest['txid'], generation + 1, prepared['content_sha256'],
             text, receipt_digest))
        _commit(connection)
        return receipt
    except BaseException:
        if connection.in_transaction:
            connection.execute('ROLLBACK')
        raise
    finally:
        connection.close()


def _existing_path(db_path):
    path = Path(db_path)
    if path.is_symlink():
        raise ValueError('store cannot be a symlink')
    if path.exists() and not path.is_file():
        raise ValueError('store path is not a regular file')
    return path


def _exists(db_path):
    """True only when a committed store is present.

    A crash can leave a database file with only a WAL and no committed tables;
    that file is treated as an absent generation-0 store and the next apply may
    create it from scratch.  A file that already holds other tables is not a
    store and is rejected instead of being adopted.
    """
    path = _existing_path(db_path)
    if not path.exists() or path.stat().st_size == 0:
        return False
    connection = sqlite3.connect(str(path), timeout=_BUSY_SECONDS,
                                 isolation_level=None)
    try:
        connection.execute('PRAGMA query_only=ON')
        tables = _tables(connection)
    finally:
        connection.close()
    if 'meta' in tables:
        return True
    if tables:
        raise ValueError('not a forecast fact store')
    return False


def _fresh_check(manifest):
    """Structural checks for a generation-0 store before its file exists."""
    if manifest['expected_generation'] != 0:
        raise ValueError(
            f"expected generation {manifest['expected_generation']} but "
            'store is at generation 0')
    if manifest['drops']:
        raise ValueError('cannot drop unknown report: '
                         + ','.join(sorted(manifest['drops'])))


def _state_checks(connection, manifest, prepared):
    """Validate drops, name collisions, ledger replay and the generation."""
    prior = connection.execute(
        'SELECT content_sha256, receipt FROM ledger WHERE txid=?',
        (manifest['txid'],)).fetchone()
    if prior is not None:
        if prior['content_sha256'] != prepared['content_sha256']:
            raise ValueError('txid already applied with different content')
        return 'replay', json.loads(prior['receipt'])
    generation = _generation(connection)
    if generation > _MAX_GENERATION:
        raise ValueError('store generation is exhausted')
    if generation != manifest['expected_generation']:
        raise ValueError(
            f"expected generation {manifest['expected_generation']} but "
            f'store is at generation {generation}')
    names = {row[0] for row in connection.execute('SELECT name FROM reports')}
    missing = sorted(set(manifest['drops']) - names)
    if missing:
        raise ValueError('cannot drop unknown report: ' + ','.join(missing))
    taken = sorted({entry['name'] for entry in prepared['catalog']} & names)
    if taken:
        raise ValueError('report name already bound: ' + ','.join(taken))
    return 'apply', None


def _commit(connection):
    # Test-only crash injection: die after the writes but before COMMIT, which
    # simulates a process killed mid-transaction.  The OS drops the connection,
    # SQLite rolls the open transaction back, and nothing reaches disk.
    if os.environ.get('DEMAND_STORE_FAULT') == 'before_commit':
        os._exit(97)
    connection.execute('COMMIT')
    # Test-only crash injection: die immediately after COMMIT returns, before
    # the caller can see or store the receipt.
    if os.environ.get('DEMAND_STORE_FAULT') == 'after_commit':
        os._exit(98)


def _mutate(connection, manifest, prepared):
    """Perform the batch inside the caller's transaction; return counts."""
    drops = tuple(sorted(manifest['drops']))
    removed = 0
    if drops:
        marks = ','.join('?' * len(drops))
        found = {row[0] for row in connection.execute(
            f'SELECT name FROM reports WHERE name IN ({marks})', drops)}
        missing = sorted(set(drops) - found)
        if missing:
            raise ValueError('cannot drop unknown report: ' + ','.join(missing))
        # Release ownership first, then delete only facts that lost their last
        # owner.  Facts still owned by an untouched report survive.
        connection.execute(
            f'DELETE FROM fact_reports WHERE name IN ({marks})', drops)
        removed = connection.execute(
            'DELETE FROM facts WHERE NOT EXISTS (SELECT 1 FROM fact_reports r'
            ' WHERE r.series=facts.series AND r.origin=facts.origin'
            ' AND r.target_date=facts.target_date AND r.method=facts.method)'
        ).rowcount
        connection.execute(
            f'DELETE FROM reports WHERE name IN ({marks})', drops)

    taken = sorted({entry['name'] for entry in prepared['catalog']}
                   & {row[0] for row in connection.execute(
                       'SELECT name FROM reports')})
    if taken:
        raise ValueError('report name already bound: ' + ','.join(taken))

    rows_in = [(key[0], key[1], key[2], key[3], values[0], _text(values[1]),
                _text(values[2]), _text(values[3]))
               for key, (values, _) in prepared['merged'].items()]
    connection.execute('DROP TABLE IF EXISTS temp.incoming')
    connection.execute(
        'CREATE TEMP TABLE incoming ('
        ' series TEXT NOT NULL, origin TEXT NOT NULL,'
        ' target_date TEXT NOT NULL, method TEXT NOT NULL,'
        ' horizon INTEGER NOT NULL, forecast TEXT NOT NULL,'
        ' actual TEXT NOT NULL, scale TEXT NOT NULL,'
        ' PRIMARY KEY (series, origin, target_date, method)) WITHOUT ROWID')
    connection.executemany(
        'INSERT INTO incoming VALUES (?, ?, ?, ?, ?, ?, ?, ?)', rows_in)
    # A put may only add facts or share a key whose values are identical; it
    # may never silently rewrite a fact still owned by an untouched report.
    # Conflicts are judged after the drops above, so a batch can replace every
    # owner of a key and change its values in the same transaction.
    clash = connection.execute(
        'SELECT i.series, i.origin, i.target_date, i.method FROM incoming i'
        ' JOIN facts f ON f.series=i.series AND f.origin=i.origin'
        ' AND f.target_date=i.target_date AND f.method=i.method'
        ' WHERE i.horizon<>f.horizon OR i.forecast<>f.forecast'
        ' OR i.actual<>f.actual OR i.scale<>f.scale').fetchone()
    if clash is not None:
        raise ValueError('forecast fact conflicts with stored values: '
                         + '/'.join(clash[:4]))
    before = connection.execute('SELECT COUNT(*) FROM facts').fetchone()[0]
    connection.execute(
        'INSERT OR IGNORE INTO facts SELECT * FROM incoming')
    after = connection.execute('SELECT COUNT(*) FROM facts').fetchone()[0]
    added = after - before
    shared = len(rows_in) - added
    connection.executemany(
        'INSERT INTO reports VALUES (?, ?, ?, ?, ?)',
        [(entry['name'], entry['kind'], entry['source_sha256'],
          entry['report_sha256'], entry['fact_count'])
         for entry in prepared['catalog']])
    ownership = []
    for name, local in prepared['incoming']:
        ownership.extend((key[0], key[1], key[2], key[3], name)
                         for key in local)
        if len(ownership) >= _BATCH_ROWS:
            connection.executemany(
                'INSERT INTO fact_reports VALUES (?, ?, ?, ?, ?)', ownership)
            ownership = []
    if ownership:
        connection.executemany(
            'INSERT INTO fact_reports VALUES (?, ?, ?, ?, ?)', ownership)
    total = connection.execute('SELECT COUNT(*) FROM facts').fetchone()[0]
    return {'removed': removed, 'shared': shared, 'added': added,
            'total': total}


def query_facts(db_path):
    """Read generation, reports, sorted facts and provenance from one snapshot."""
    connection = _read_snapshot(db_path)
    try:
        connection.execute('BEGIN')
        tables = _tables(connection)
        if 'meta' not in tables:
            raise ValueError('not a supported forecast fact store')
        schema = connection.execute(
            "SELECT value FROM meta WHERE key='schema_version'").fetchone()
        if schema is None or schema[0] != '1':
            raise ValueError('unsupported store schema_version')
        generation = _generation(connection)
        catalog = _catalog(connection)
        facts, current = [], None
        cursor = connection.execute(
            'SELECT f.series, f.origin, f.target_date, f.method, f.horizon,'
            ' f.forecast, f.actual, f.scale, r.name'
            ' FROM facts f JOIN fact_reports r ON'
            ' r.series=f.series AND r.origin=f.origin'
            ' AND r.target_date=f.target_date AND r.method=f.method'
            ' ORDER BY f.series, f.origin, f.target_date, f.method, r.name')
        for row in cursor:
            key = (row['series'], row['origin'], row['target_date'],
                   row['method'])
            if current is None or current[0] != key:
                current = (key, {'series': key[0], 'origin': key[1],
                                 'target_date': key[2], 'method': key[3],
                                 'horizon': row['horizon'],
                                 'forecast': row['forecast'],
                                 'actual': row['actual'],
                                 'scale': row['scale'], 'reports': []})
                facts.append(current[1])
            current[1]['reports'].append(row['name'])
        classified = connection.execute(
            'SELECT COUNT(*) FROM facts f WHERE NOT EXISTS ('
            ' SELECT 1 FROM fact_reports r WHERE ' + _KEY_JOIN + ')'
        ).fetchone()[0]
        if classified:
            raise ValueError('store is internally inconsistent')
        result = {'schema_version': 1, 'mode': 'forecast_store_query',
                  'generation': generation, 'reports': catalog,
                  'facts': facts, 'fact_count': len(facts)}
        connection.execute('COMMIT')
        return result
    except BaseException:
        if connection.in_transaction:
            connection.execute('ROLLBACK')
        raise
    finally:
        connection.close()
