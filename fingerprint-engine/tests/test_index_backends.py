"""Per-backend CRUD, transaction hygiene, WAL, concurrency, schema-version load."""

from __future__ import annotations

import json
import sqlite3
import sys
import threading
from pathlib import Path

import pytest
from _fixtures import (
    PG_DSN,
    fp_with_hashes,
    make_fingerprint,
    requires_pg,
)
from _fixtures import (
    fake_redis as _fake_redis,
)

from fingerprint_engine.core.exceptions import (
    FingerprintError,
    InvalidSnapshotError,
)
from fingerprint_engine.core.index import (
    SNAPSHOT_SCHEMA_VERSION,
    InMemoryHashIndex,
    PostgresHashIndex,
    RedisHashIndex,
    SQLiteHashIndex,
)
from fingerprint_engine.core.models import (
    ConstellationHash,
    Fingerprint,
)


def test_constellation_hash_rejects_out_of_range_code() -> None:
    # A4 (model boundary): a hash code outside the unsigned-64 range cannot be
    # constructed, so every backend rejects the same input identically instead
    # of InMemory accepting it while the SQL backends overflow their column.
    from fingerprint_engine.core.models import _HASH_CODE_MAX

    # In-range codes (including the boundary) construct fine.
    for ok in (0, 1, _HASH_CODE_MAX):
        ConstellationHash(hash_code=ok, time_offset=0, anchor_time=0,
                          target_time=0, freq1=0, freq2=0, delta_t=0)
    for bad in (-1, _HASH_CODE_MAX + 1, 2**65):
        with pytest.raises(ValueError, match="unsigned 64-bit"):
            ConstellationHash(hash_code=bad, time_offset=0, anchor_time=0,
                              target_time=0, freq1=0, freq2=0, delta_t=0)


class _PostingInsertFails:
    """Connection proxy that fails the postings INSERT, to exercise add() rollback."""

    def __init__(self, real: sqlite3.Connection) -> None:
        self._real = real

    def executemany(self, sql: str, params: object):  # noqa: ANN001 - test proxy
        if "INSERT INTO postings" in sql:
            raise sqlite3.OperationalError("simulated posting insert failure")
        return self._real.executemany(sql, params)

    def __getattr__(self, name: str) -> object:
        return getattr(self._real, name)


def test_sqlite_add_rolls_back_on_posting_failure(tmp_path: Path) -> None:
    # A4 (rollback): a failure after the files INSERT must not leave a committed
    # phantom file (zero postings) that the next successful commit flushes.
    index = SQLiteHashIndex(str(tmp_path / "idx.sqlite3"))
    real_conn = index._conn
    index._conn = _PostingInsertFails(real_conn)  # type: ignore[assignment]
    with pytest.raises(sqlite3.OperationalError):
        index.add(make_fingerprint("file-a", [1, 2, 3]))
    index._conn = real_conn  # restore for assertions

    # No phantom row survived the failed add, and a later good add is unpolluted.
    assert index.file_count == 0
    assert not index.contains("file-a")
    index.add(make_fingerprint("file-b", [4, 5]))
    assert index.file_count == 1
    assert index.contains("file-b")
    assert index.search(make_fingerprint("file-b", [4, 5]))[0].file_id == "file-b"


def test_redis_mutators_run_under_the_per_index_lock() -> None:
    # A5: fakeredis serializes commands so the read-modify-write race cannot be
    # reproduced in-env; assert directly that add()/remove() acquire the per-index
    # RLock (the same in-process concurrency guard the SQL backends use). add()'s
    # nested remove() re-enters the re-entrant lock, so add() acquires twice.
    index = RedisHashIndex(client=_fake_redis())
    assert isinstance(index._lock, type(threading.RLock()))

    class _CountingLock:
        def __init__(self, real: object) -> None:
            self._real = real
            self.enters = 0

        def __enter__(self) -> object:
            self.enters += 1
            return self._real.__enter__()

        def __exit__(self, *args: object) -> object:
            return self._real.__exit__(*args)

    counting = _CountingLock(index._lock)
    index._lock = counting  # type: ignore[assignment]
    index.add(make_fingerprint("file-a", [1, 2, 3]))
    assert counting.enters >= 2  # add() + its nested remove()
    before = counting.enters
    index.remove("file-a")
    assert counting.enters > before
    # The lock did not change behaviour: the add/remove round-trip still works.
    assert index.file_count == 0


def test_redis_backend_search_matches_in_memory() -> None:
    redis_index = RedisHashIndex(client=_fake_redis(), key_prefix="t1")
    mem_index = InMemoryHashIndex()
    for file_id, offsets in [("aligned", [10, 20, 30, 40]), ("scattered", [2, 40, 99, 125])]:
        fingerprint = make_fingerprint(file_id, offsets)
        redis_index.add(fingerprint)
        mem_index.add(fingerprint)
    query = make_fingerprint("query", [3, 13, 23, 33])

    redis_results = redis_index.search(query, top_k=2)
    mem_results = mem_index.search(query, top_k=2)

    # Identical ranking and scores: the search logic is shared in the base class.
    assert [r.file_id for r in redis_results] == [r.file_id for r in mem_results]
    assert redis_results[0].file_id == "aligned"
    assert redis_results[0].aligned_votes == 4
    assert redis_results[0].offset == 7
    assert redis_results[0].score == mem_results[0].score
    assert redis_index.file_count == 2
    assert redis_index.posting_count == 8


def test_redis_backend_remove_and_replace() -> None:
    index = RedisHashIndex(client=_fake_redis(), key_prefix="t2")
    index.add(make_fingerprint("a", [1, 2, 3]))
    index.add(make_fingerprint("b", [1, 2]))
    assert index.file_count == 2
    assert index.posting_count == 5

    # Re-adding the same file_id replaces it (no double counting).
    index.add(make_fingerprint("a", [9]))
    assert index.file_count == 2
    assert index.posting_count == 3  # a:1 + b:2

    index.remove("b")
    assert index.file_count == 1
    assert index.posting_count == 1
    assert all(posting.file_id != "b" for posting in index.query(1000))


def test_redis_snapshot_interops_with_in_memory(tmp_path: Path) -> None:
    snapshot = tmp_path / "snap.json"
    fingerprint = make_fingerprint("file-a", [1, 2, 3])

    source = RedisHashIndex(client=_fake_redis(), key_prefix="t3")
    source.add(fingerprint)
    source.save(snapshot)

    # Redis snapshot loads into the in-memory backend...
    loaded = InMemoryHashIndex.load(snapshot)
    assert loaded.file_count == 1
    assert loaded.posting_count == 3
    assert loaded.search(fingerprint)[0].file_id == "file-a"

    # ...and any snapshot bulk-loads back into a fresh Redis index.
    target = RedisHashIndex(client=_fake_redis(), key_prefix="t4").load_snapshot(snapshot)
    assert target.file_count == 1
    assert target.posting_count == 3
    assert target.search(fingerprint)[0].file_id == "file-a"


def test_sqlite_backend_search_matches_in_memory() -> None:
    sqlite_index = SQLiteHashIndex(":memory:")
    mem_index = InMemoryHashIndex()
    for file_id, offsets in [("aligned", [10, 20, 30, 40]), ("scattered", [2, 40, 99, 125])]:
        fingerprint = make_fingerprint(file_id, offsets)
        sqlite_index.add(fingerprint)
        mem_index.add(fingerprint)
    query = make_fingerprint("query", [3, 13, 23, 33])

    sqlite_results = sqlite_index.search(query, top_k=2)
    mem_results = mem_index.search(query, top_k=2)

    assert [r.file_id for r in sqlite_results] == [r.file_id for r in mem_results]
    assert sqlite_results[0].file_id == "aligned"
    assert sqlite_results[0].aligned_votes == 4
    assert sqlite_results[0].offset == 7
    assert sqlite_results[0].score == mem_results[0].score
    assert sqlite_index.file_count == 2
    assert sqlite_index.posting_count == 8


def test_sqlite_backend_remove_and_replace() -> None:
    index = SQLiteHashIndex(":memory:")
    index.add(make_fingerprint("a", [1, 2, 3]))
    index.add(make_fingerprint("b", [1, 2]))
    assert index.file_count == 2
    assert index.posting_count == 5

    index.add(make_fingerprint("a", [9]))  # replace, no double counting
    assert index.file_count == 2
    assert index.posting_count == 3

    index.remove("b")
    assert index.file_count == 1
    assert index.posting_count == 1
    assert all(posting.file_id != "b" for posting in index.query(1000))


def test_sqlite_snapshot_interops_with_in_memory(tmp_path: Path) -> None:
    snapshot = tmp_path / "snap.json"
    fingerprint = make_fingerprint("file-a", [1, 2, 3])

    source = SQLiteHashIndex(":memory:")
    source.add(fingerprint)
    source.save(snapshot)

    loaded = InMemoryHashIndex.load(snapshot)
    assert loaded.file_count == 1
    assert loaded.posting_count == 3
    assert loaded.search(fingerprint)[0].file_id == "file-a"

    target = SQLiteHashIndex(":memory:").load_snapshot(snapshot)
    assert target.file_count == 1
    assert target.posting_count == 3
    assert target.search(fingerprint)[0].file_id == "file-a"


def test_sqlite_handles_full_64bit_hash_codes() -> None:
    # Regression: hash codes are unsigned 64-bit but SQLite INTEGER is signed
    # 64-bit; a code >= 2**63 must round-trip via the signed-offset mapping.
    big = (1 << 64) - 1
    fingerprint = Fingerprint(
        file_id="big",
        path="/tmp/big",
        handler="test",
        size_bytes=1,
        content_sha256="big",
        config={},
        hashes=[
            ConstellationHash(
                hash_code=big, time_offset=4, anchor_time=4,
                target_time=5, freq1=1, freq2=2, delta_t=1,
            )
        ],
        metadata={},
    )
    index = SQLiteHashIndex(":memory:")
    index.add(fingerprint)

    assert index.posting_count == 1
    postings = index.query(big)
    assert postings and postings[0].hash_code == big
    assert index.search(fingerprint)[0].file_id == "big"
    assert index.to_dict()["files"]["big"][0][0] == big  # snapshot keeps the value


def test_postgres_signed_offset_roundtrip_is_overflow_safe() -> None:
    # Serverless: PostgreSQL BIGINT is signed 64-bit, so unsigned 64-bit hash
    # codes must map reversibly into signed range (same lesson as SQLite).
    extremes = [0, 1, 1 << 62, (1 << 63) - 1, 1 << 63, (1 << 64) - 1]
    for code in extremes:
        assert PostgresHashIndex._decode(PostgresHashIndex._encode(code)) == code
        assert -(1 << 63) <= PostgresHashIndex._encode(code) < (1 << 63)  # fits BIGINT


@requires_pg
def test_postgres_backend_search_matches_in_memory(pg_index) -> None:
    mem_index = InMemoryHashIndex()
    for file_id, offsets in [("aligned", [10, 20, 30, 40]), ("scattered", [2, 40, 99, 125])]:
        fingerprint = make_fingerprint(file_id, offsets)
        pg_index.add(fingerprint)
        mem_index.add(fingerprint)
    query = make_fingerprint("query", [3, 13, 23, 33])

    pg_results = pg_index.search(query, top_k=2)
    mem_results = mem_index.search(query, top_k=2)

    assert [r.file_id for r in pg_results] == [r.file_id for r in mem_results]
    assert pg_results[0].file_id == "aligned"
    assert pg_results[0].aligned_votes == 4
    assert pg_results[0].offset == 7
    assert pg_results[0].score == mem_results[0].score
    assert pg_index.file_count == 2
    assert pg_index.posting_count == 8


@requires_pg
def test_postgres_backend_remove_and_replace(pg_index) -> None:
    pg_index.add(make_fingerprint("a", [1, 2, 3]))
    pg_index.add(make_fingerprint("b", [1, 2]))
    assert pg_index.file_count == 2
    assert pg_index.posting_count == 5

    pg_index.add(make_fingerprint("a", [9]))  # replace, no double counting
    assert pg_index.file_count == 2
    assert pg_index.posting_count == 3

    pg_index.remove("b")
    assert pg_index.file_count == 1
    assert pg_index.posting_count == 1
    assert all(posting.file_id != "b" for posting in pg_index.query(1000))


@requires_pg
def test_postgres_handles_full_64bit_hash_codes(pg_index) -> None:
    big = (1 << 64) - 1
    fingerprint = Fingerprint(
        file_id="big", path="/tmp/big", handler="test", size_bytes=1,
        content_sha256="big", config={},
        hashes=[ConstellationHash(hash_code=big, time_offset=4, anchor_time=4,
                                  target_time=5, freq1=1, freq2=2, delta_t=1)],
        metadata={},
    )
    pg_index.add(fingerprint)
    assert pg_index.query(big)[0].hash_code == big
    assert pg_index.search(fingerprint)[0].file_id == "big"
    assert pg_index.to_dict()["files"]["big"][0][0] == big


@requires_pg
def test_postgres_snapshot_interops_with_in_memory(pg_index, tmp_path: Path) -> None:
    snapshot = tmp_path / "snap.json"
    fingerprint = make_fingerprint("file-a", [1, 2, 3])
    pg_index.add(fingerprint)
    pg_index.save(snapshot)

    loaded = InMemoryHashIndex.load(snapshot)
    assert loaded.file_count == 1
    assert loaded.posting_count == 3
    assert loaded.search(fingerprint)[0].file_id == "file-a"


@requires_pg
def test_postgres_read_paths_do_not_leave_idle_in_transaction(pg_index) -> None:
    # psycopg opens a transaction on the first execute; a pure read that never
    # commits/rolls back leaves the connection idle-in-transaction, holding
    # locks. After each read path the connection must be back to IDLE.
    import psycopg

    pg_index.add(make_fingerprint("aligned", [10, 20, 30, 40]))

    def assert_idle() -> None:
        assert pg_index._conn.info.transaction_status == psycopg.pq.TransactionStatus.IDLE

    assert pg_index.file_count == 1
    assert_idle()
    assert pg_index.posting_count == 4
    assert_idle()
    pg_index.query(1000)
    assert_idle()
    pg_index.query_many([1000, 1001])
    assert_idle()
    pg_index._metadata_for("aligned")
    assert_idle()
    results = pg_index.search(make_fingerprint("query", [3, 13, 23, 33]), top_k=1)
    assert results[0].file_id == "aligned"  # scoring/tie-break unchanged
    assert_idle()


@requires_pg
def test_postgres_context_manager_closes_connection() -> None:
    # Own index (not the shared pg_index fixture) so closing the connection here
    # does not break fixture teardown.
    import psycopg

    index = PostgresHashIndex(dsn=PG_DSN, table_prefix="fp_pytest_cm")
    with index as entered:
        assert entered is index  # __enter__ returns self
    assert index._conn.closed  # __exit__ closed the connection
    with pytest.raises(psycopg.OperationalError):
        index.query(1000)


def _stop_hash_corpus(index):
    # Code 7 appears in all 4 files (a "stop" hash); other codes are file-unique.
    index.add(fp_with_hashes("a", [(7, 0), (100, 1), (101, 2)]))
    index.add(fp_with_hashes("b", [(7, 0), (200, 1), (201, 2)]))
    index.add(fp_with_hashes("c", [(7, 0), (300, 1), (301, 2)]))
    index.add(fp_with_hashes("d", [(7, 0), (400, 1), (401, 2)]))
    return index


def test_prune_stop_hashes_removes_common_codes_and_recalibrates() -> None:
    for index in (InMemoryHashIndex(), SQLiteHashIndex(":memory:")):
        _stop_hash_corpus(index)
        assert index.posting_count == 12
        assert len(index.query(7)) == 4  # common code present in all files

        removed = index.prune_stop_hashes(max_df_ratio=0.5)  # code 7 is in 100% of files

        assert removed == 4
        assert index.query(7) == []                 # stop code gone
        assert index.posting_count == 8
        assert len(index.query(100)) == 1           # discriminative codes kept

        # A self-search still matches at full confidence: the pruned target's
        # hash_count was recalibrated, so aligned / target == 1.0.
        result = index.search(fp_with_hashes("a", [(7, 0), (100, 1), (101, 2)]), top_k=1)[0]
        assert result.file_id == "a"
        assert result.confidence == 1.0
        assert result.metadata["hash_count"] == 2   # was 3, code 7 pruned


def test_redis_prune_stop_hashes_declines() -> None:
    index = RedisHashIndex(client=_fake_redis(), key_prefix="t9")
    with pytest.raises(NotImplementedError):
        index.prune_stop_hashes()


def test_sqlite_file_backend_persists(tmp_path: Path) -> None:
    # A file-backed SQLite index must survive being reopened.
    db = tmp_path / "index.sqlite3"
    fingerprint = make_fingerprint("persisted", [5, 6, 7])
    SQLiteHashIndex(db).add(fingerprint)

    reopened = SQLiteHashIndex(db)
    assert reopened.file_count == 1
    assert reopened.posting_count == 3
    assert reopened.search(fingerprint)[0].file_id == "persisted"


def test_sqlite_search_does_not_leave_open_write_transaction() -> None:
    # Regression: _aggregate stages query pairs with CREATE TEMP/DELETE/INSERT
    # (DML), which opens an implicit write transaction. Without a commit the
    # connection stays in a transaction holding a write lock for its lifetime,
    # blocking other writers cross-process. A read-only search must commit/close
    # that transaction so in_transaction is False on return.
    index = SQLiteHashIndex(":memory:")
    index.add(make_fingerprint("aligned", [10, 20, 30, 40]))

    results = index.search(make_fingerprint("query", [3, 13, 23, 33]), top_k=1)

    assert results[0].file_id == "aligned"  # scoring/tie-break unchanged
    assert index._conn.in_transaction is False  # no lingering write transaction


def test_sqlite_context_manager_closes_connection() -> None:
    # The context-manager protocol must give deterministic cleanup: __exit__
    # closes the underlying connection so subsequent use raises.
    with SQLiteHashIndex(":memory:") as index:
        assert index.__enter__() is index  # __enter__ returns self
        index.add(make_fingerprint("a", [1, 2, 3]))
        conn = index._conn

    with pytest.raises(sqlite3.ProgrammingError):
        conn.execute("SELECT 1")  # connection is closed after the with block


def test_save_stamps_schema_version_and_round_trips(tmp_path: Path) -> None:
    # save() must stamp the current schema_version into the snapshot, and a
    # snapshot carrying that version must load back intact.
    index_path = tmp_path / "index.json"
    index = InMemoryHashIndex()
    fingerprint = make_fingerprint("file-a", [1, 2, 3])
    index.add(fingerprint)
    index.save(index_path)

    raw = json.loads(index_path.read_text(encoding="utf-8"))
    assert raw["schema_version"] == SNAPSHOT_SCHEMA_VERSION

    loaded = InMemoryHashIndex.load(index_path)
    assert loaded.file_count == 1
    assert loaded.posting_count == 3
    assert loaded.search(fingerprint)[0].file_id == "file-a"


def test_load_rejects_unsupported_schema_version(tmp_path: Path) -> None:
    # An explicit, unsupported schema_version must raise InvalidSnapshotError
    # (which is also a ValueError) on every load entry point.
    index_path = tmp_path / "index.json"
    payload = {
        "backend": "in_memory",
        "files": {"file-a": [[1000, 1]]},
        "metadata": {},
        "schema_version": 9999,
    }
    index_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(InvalidSnapshotError, match="schema_version"):
        InMemoryHashIndex.load(index_path)
    # Joins both families: ValueError (back-compat) and FingerprintError.
    assert issubclass(InvalidSnapshotError, ValueError)
    assert issubclass(InvalidSnapshotError, FingerprintError)
    with pytest.raises(ValueError):
        InMemoryHashIndex.from_dict(payload)
    with pytest.raises(InvalidSnapshotError, match="schema_version"):
        SQLiteHashIndex(":memory:").load_snapshot(index_path)


def test_load_accepts_absent_schema_version_as_legacy(tmp_path: Path) -> None:
    # Snapshots written before versioning have NO schema_version key; they must
    # still load (treated as version 1), not be rejected.
    index_path = tmp_path / "index.json"
    payload = {
        "backend": "in_memory",
        "files": {"file-a": [[1000, 1], [1001, 2]]},
        "metadata": {"file-a": {"handler": "test", "hash_count": 2}},
    }
    assert "schema_version" not in payload
    index_path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = InMemoryHashIndex.load(index_path)
    assert loaded.file_count == 1
    assert loaded.posting_count == 2


def test_load_raises_when_primary_missing_and_backup_corrupt(tmp_path: Path) -> None:
    # The primary is gone but a .bak exists and is itself corrupt: this must
    # surface as InvalidSnapshotError, not a raw JSONDecodeError.
    index_path = tmp_path / "index.json"
    backup = index_path.with_name("index.json.bak")
    backup.write_text('{"backend": "in_memory", "files": {"file-a": [[100', encoding="utf-8")
    assert not index_path.exists()

    with pytest.raises(InvalidSnapshotError):
        InMemoryHashIndex.load(index_path)


def test_load_skips_out_of_range_hash_codes_without_overflow(tmp_path: Path) -> None:
    # A hash_code outside the unsigned 64-bit range must be skipped on load
    # rather than aborting the whole cross-backend import with OverflowError
    # deep inside the SQL signed-offset encode.
    index_path = tmp_path / "index.json"
    too_big = 1 << 64          # one past the unsigned 64-bit max
    negative = -1              # below the unsigned 64-bit min
    in_range = (1 << 64) - 1   # the largest legal code
    payload = {
        "backend": "in_memory",
        "files": {"file-a": [[in_range, 1], [too_big, 2], [negative, 3]]},
        "metadata": {"file-a": {"handler": "test"}},
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
    }
    index_path.write_text(json.dumps(payload), encoding="utf-8")

    # In-memory load keeps only the in-range posting.
    mem = InMemoryHashIndex.load(index_path)
    assert mem.posting_count == 1
    assert mem.query(in_range) and mem.query(in_range)[0].hash_code == in_range
    assert mem.query(too_big) == []

    # The cross-backend bulk load into SQLite must NOT raise OverflowError.
    sqlite_index = SQLiteHashIndex(":memory:").load_snapshot(index_path)
    assert sqlite_index.posting_count == 1
    assert sqlite_index.query(in_range)[0].hash_code == in_range


def test_sqlite_file_backend_uses_wal_journal_mode(tmp_path: Path) -> None:
    # A file-backed SQLite index must enable WAL after init so concurrent
    # readers can run alongside one writer.
    db = tmp_path / "index.sqlite3"
    index = SQLiteHashIndex(db)
    mode = index._conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode.lower() == "wal"
    # busy_timeout is set so a contended writer waits rather than failing fast.
    assert int(index._conn.execute("PRAGMA busy_timeout").fetchone()[0]) == 5000


def test_in_memory_concurrent_add_and_search_stay_consistent() -> None:
    # Many concurrent add()/search() calls on one InMemoryHashIndex must not
    # crash or corrupt counts: writes are lock-serialized, reads are GIL-safe.
    index = InMemoryHashIndex()
    file_ids = [f"file-{i}" for i in range(50)]
    errors: list[BaseException] = []
    barrier = threading.Barrier(len(file_ids))

    def worker(file_id: str) -> None:
        try:
            barrier.wait()
            index.add(make_fingerprint(file_id, [1, 2, 3]))
            # Concurrent reads must never observe a half-applied write.
            index.search(make_fingerprint("query", [1, 2, 3]), top_k=5)
        except BaseException as exc:  # noqa: BLE001 - surface any thread failure
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(fid,)) for fid in file_ids]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors
    assert index.file_count == len(file_ids)
    # Each file contributed exactly 3 postings; no double-counting or loss.
    assert index.posting_count == 3 * len(file_ids)
    assert {r.file_id for r in index.search(make_fingerprint("q", [1, 2, 3]), top_k=100)} == set(file_ids)


def _fp_codes(file_id: str, codes: list[int]) -> Fingerprint:
    """A fingerprint with EXPLICIT hash codes (so two files can be disjoint)."""

    hashes = [
        ConstellationHash(
            hash_code=code,
            time_offset=index,
            anchor_time=index,
            target_time=index + 1,
            freq1=0,
            freq2=0,
            delta_t=1,
        )
        for index, code in enumerate(codes)
    ]
    return Fingerprint(
        file_id=file_id,
        path=f"/tmp/{file_id}",
        handler="test",
        size_bytes=len(codes),
        content_sha256=file_id,
        config={},
        hashes=hashes,
    )


def test_sqlite_concurrent_search_is_consistent_and_uncorrupted() -> None:
    # Regression for the connection-global ``_query`` TEMP TABLE: concurrent
    # search() calls on one SQLite-backed index (exactly what the FastAPI
    # threadpool issues against the shared active_index) must not interleave the
    # DELETE/INSERT/SELECT steps into an InterfaceError or cross-contaminated
    # rankings. Two files with DISJOINT hash spaces: a query that overlaps only
    # one must NEVER return the other. Without the @_synchronized serialization
    # this raises and/or returns the wrong file (reproduced live in review).
    index = SQLiteHashIndex(":memory:")
    index.add(_fp_codes("A", list(range(1000, 1100))))
    index.add(_fp_codes("B", list(range(2000, 2100))))

    errors: list[BaseException] = []
    contaminated: list[tuple[str, str]] = []
    n = 16
    barrier = threading.Barrier(n)

    def worker(which: str) -> None:
        try:
            barrier.wait()
            codes = list(range(1000, 1100)) if which == "A" else list(range(2000, 2100))
            for _ in range(25):
                results = index.search(_fp_codes("q", codes), top_k=5)
                if results and results[0].file_id != which:
                    contaminated.append((which, results[0].file_id))
        except BaseException as exc:  # noqa: BLE001 - surface any thread failure
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=("A" if i % 2 else "B",)) for i in range(n)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors, errors[:1]
    assert not contaminated, contaminated[:3]


def test_in_memory_concurrent_readd_remove_never_keyerrors_readers() -> None:
    # Reproduces the surrogate-deref concurrency regression: lock-free readers
    # (search()/query()) read a posting list and THEN dereference the surrogate
    # -> file_id map. With delete-on-remove, a concurrent remove()/re-add() of an
    # OVERLAPPING id set retires that surrogate in between, so the reader hits a
    # KeyError. The fix keeps surrogate mappings alive across remove(), so the
    # deref always resolves. Writers churn an overlapping id set under contention
    # while readers search()/query() the same hash codes; assert NO reader raises.
    index = InMemoryHashIndex()
    # Overlapping id set: writers add/remove/re-add the SAME pool of ids, so a
    # surrogate a reader just observed is the one a writer is retiring/replacing.
    shared_ids = [f"shared-{i}" for i in range(12)]
    # The query hash codes (1000, 1001, 1002) match make_fingerprint's codes, so
    # every reader fetch touches the churning postings -- maximal deref pressure.
    query = make_fingerprint("q", [0, 1, 2])
    hash_codes = [item.hash_code for item in query.hashes]

    n_writers = 4
    n_readers = 4
    iterations = 300
    errors: list[BaseException] = []
    barrier = threading.Barrier(n_writers + n_readers)
    stop = threading.Event()

    def writer(seed: int) -> None:
        try:
            barrier.wait()
            for i in range(iterations):
                fid = shared_ids[(seed + i) % len(shared_ids)]
                index.add(make_fingerprint(fid, [0, 1, 2]))
                index.remove(fid)
                index.add(make_fingerprint(fid, [0, 1, 2]))  # re-add same id
        except BaseException as exc:  # noqa: BLE001 - surface any thread failure
            errors.append(exc)
        finally:
            stop.set()

    def reader() -> None:
        try:
            barrier.wait()
            # Spin reads until writers finish; a KeyError here is the regression.
            while not stop.is_set():
                index.search(query, top_k=10)
                for code in hash_codes:
                    index.query(code)
        except BaseException as exc:  # noqa: BLE001 - surface any thread failure
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(s,)) for s in range(n_writers)]
    threads += [threading.Thread(target=reader) for _ in range(n_readers)]
    # Force frequent thread switches so the (GIL-protected but multi-bytecode)
    # window between a reader capturing a posting list and dereferencing its
    # surrogate is reliably interleaved with a writer's remove(). On the buggy
    # delete-on-remove code this surfaces the KeyError every run; restored in
    # finally so the tight interval never leaks into other tests.
    previous_interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    finally:
        sys.setswitchinterval(previous_interval)

    # No reader (or writer) raised -- in particular no surrogate-deref KeyError.
    assert not errors, errors
    # Final state is self-consistent: every surviving file has live postings and
    # posting_count equals the sum over surviving files (no orphaned/lost rows).
    surviving = index.list_files()
    expected_postings = sum(len(index._file_entries[fid]) for fid in surviving)
    assert index.posting_count == expected_postings
    assert index.file_count == len(surviving)
    # Surviving ids resolve cleanly through a final search (no stale surrogate).
    assert set(r.file_id for r in index.search(query, top_k=100)) <= set(surviving)


