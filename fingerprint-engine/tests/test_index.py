from __future__ import annotations

import json
import os
import sqlite3
import sys
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fingerprint_engine.core.exceptions import FingerprintError, InvalidSnapshotError
from fingerprint_engine.core.index import (
    SNAPSHOT_SCHEMA_VERSION,
    InMemoryHashIndex,
    PostgresHashIndex,
    RedisHashIndex,
    SQLiteHashIndex,
)
from fingerprint_engine.core.models import Calibration, ConstellationHash, Fingerprint


def _fake_redis():
    fakeredis = pytest.importorskip("fakeredis")
    return fakeredis.FakeStrictRedis(decode_responses=True)


# Postgres integration tests need a live server; set FINGERPRINT_TEST_PG_DSN to run them.
PG_DSN = os.environ.get("FINGERPRINT_TEST_PG_DSN")
requires_pg = pytest.mark.skipif(not PG_DSN, reason="set FINGERPRINT_TEST_PG_DSN to run Postgres tests")


@pytest.fixture
def pg_index():
    index = PostgresHashIndex(dsn=PG_DSN, table_prefix="fp_pytest")
    with index._conn.cursor() as cur:  # start from a clean slate
        cur.execute(f"TRUNCATE {index._files_table}, {index._postings_table}")
    index._conn.commit()
    try:
        yield index
    finally:
        with index._conn.cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS {index._postings_table}, {index._files_table}")
        index._conn.commit()
        index.close()


def make_fingerprint(file_id: str, offsets: list[int]) -> Fingerprint:
    hashes = [
        ConstellationHash(
            hash_code=1000 + index,
            time_offset=offset,
            anchor_time=offset,
            target_time=offset + 1,
            freq1=10 + index,
            freq2=20 + index,
            delta_t=1,
        )
        for index, offset in enumerate(offsets)
    ]
    return Fingerprint(
        file_id=file_id,
        path=f"/tmp/{file_id}",
        handler="test",
        size_bytes=10,
        content_sha256=file_id,
        config={},
        hashes=hashes,
        metadata={"label": file_id},
    )


def fp_with_hashes(file_id: str, code_offsets: list[tuple[int, int]]) -> Fingerprint:
    """Fingerprint with explicit (hash_code, time_offset) pairs."""
    hashes = [
        ConstellationHash(hash_code=code, time_offset=offset, anchor_time=offset,
                          target_time=offset + 1, freq1=1, freq2=2, delta_t=1)
        for code, offset in code_offsets
    ]
    return Fingerprint(file_id=file_id, path=f"/tmp/{file_id}", handler="test", size_bytes=10,
                       content_sha256=file_id, config={}, hashes=hashes, metadata={})


def _search_tuples(index, query: Fingerprint, top_k: int = 10) -> list[tuple[str, int, int, float]]:
    """The cross-backend-comparable shape of a ranked result list.

    Pins exactly the fields whose computation is shared in the base class
    (file_id, winning offset, aligned votes, score) and whose ORDER is the
    contract every backend must reproduce byte-identically.
    """

    return [(r.file_id, r.offset, r.aligned_votes, r.score) for r in index.search(query, top_k=top_k)]


def test_time_coherent_search_ranks_aligned_match_first() -> None:
    index = InMemoryHashIndex()
    index.add(make_fingerprint("aligned", [10, 20, 30, 40]))
    index.add(make_fingerprint("scattered", [2, 40, 99, 125]))
    query = make_fingerprint("query", [3, 13, 23, 33])

    results = index.search(query, top_k=2)

    assert results[0].file_id == "aligned"
    assert results[0].aligned_votes == 4
    assert results[0].offset == 7


def test_query_many_matches_individual_and_handles_chunking() -> None:
    # Batched lookup must equal per-code query() and survive crossing the
    # SQLite IN-chunk boundary (500). Run for the dict and SQLite backends.
    for index in (InMemoryHashIndex(), SQLiteHashIndex(":memory:")):
        index.add(make_fingerprint("big", list(range(600))))   # codes 1000..1599
        index.add(make_fingerprint("other", [0, 1, 2]))        # codes 1000..1002
        codes = list(range(1000, 1600)) + [999999]             # spans chunk + 1 absent
        batched = index.query_many(codes)

        assert set(batched) == set(codes)          # every requested code present
        assert batched[999999] == []               # absent code -> empty list
        assert {p.file_id for p in batched[1000]} == {"big", "other"}
        for code in (1000, 1300, 1599):            # parity with individual query()
            assert sorted((p.file_id, p.time_offset) for p in batched[code]) == \
                   sorted((p.file_id, p.time_offset) for p in index.query(code))


def test_search_reports_normalized_confidence() -> None:
    index = InMemoryHashIndex()
    index.add(make_fingerprint("aligned", [10, 20, 30, 40]))
    index.add(make_fingerprint("scattered", [2, 40, 99, 125]))
    query = make_fingerprint("query", [3, 13, 23, 33])

    results = {r.file_id: r for r in index.search(query, top_k=5)}

    # All four query hashes align coherently with "aligned" -> full confidence.
    assert results["aligned"].confidence == 1.0
    # "scattered" shares hashes but no coherent offset -> low confidence.
    assert 0.0 < results["scattered"].confidence < 0.5
    assert all(0.0 <= r.confidence <= 1.0 for r in results.values())


def test_calibration_filters_and_per_handler_overrides() -> None:
    index = InMemoryHashIndex()
    index.add(make_fingerprint("aligned", [10, 20, 30, 40]))
    index.add(make_fingerprint("scattered", [2, 40, 99, 125]))
    query = make_fingerprint("query", [3, 13, 23, 33])

    # A uniform threshold drops the low-confidence (incoherent) match.
    strict = index.search(query, top_k=5, calibration=Calibration(default_min_confidence=0.5))
    assert [r.file_id for r in strict] == ["aligned"]
    assert all(r.confidence >= 0.5 for r in strict)

    # A per-handler override loosens the cutoff for handler 'test', keeping both.
    loose = index.search(
        query,
        top_k=5,
        calibration=Calibration(default_min_confidence=0.9, per_handler={"test": 0.1}),
    )
    assert {r.file_id for r in loose} == {"aligned", "scattered"}


def test_index_save_and_load_round_trips(tmp_path: Path) -> None:
    index_path = tmp_path / "index.json"
    index = InMemoryHashIndex()
    fingerprint = make_fingerprint("file-a", [1, 2, 3])
    index.add(fingerprint)
    index.save(index_path)

    loaded = InMemoryHashIndex.load(index_path)
    results = loaded.search(fingerprint)

    assert loaded.file_count == 1
    assert loaded.posting_count == 3
    assert results[0].file_id == "file-a"


def test_save_is_atomic_and_recovers_from_corrupt_primary(tmp_path: Path) -> None:
    # A durable save must leave no partial temp file in the directory, and a
    # corrupt primary with a valid .bak must transparently load from the backup.
    index_path = tmp_path / "index.json"
    index = InMemoryHashIndex()
    index.add(make_fingerprint("file-a", [1, 2, 3]))
    index.save(index_path)

    # No stray temp file left behind by the atomic write.
    assert [p.name for p in tmp_path.iterdir()] == ["index.json"]

    # Simulate a good backup next to a primary that was truncated mid-write.
    backup = index_path.with_name("index.json.bak")
    backup.write_text(index_path.read_text(encoding="utf-8"), encoding="utf-8")
    index_path.write_text('{"backend": "in_memory", "files": {"file-a": [[100', encoding="utf-8")

    loaded = InMemoryHashIndex.load(index_path)

    assert loaded.file_count == 1
    assert loaded.posting_count == 3


def test_save_keeps_backup_of_prior_contents(tmp_path: Path) -> None:
    # The second save must preserve the first save's snapshot at <dest>.bak so a
    # later corrupt primary can fall back to the previous good state.
    index_path = tmp_path / "index.json"
    backup = index_path.with_name("index.json.bak")

    first = InMemoryHashIndex()
    first.add(make_fingerprint("file-a", [1, 2, 3]))
    first.save(index_path)
    first_contents = index_path.read_text(encoding="utf-8")
    assert not backup.exists()  # nothing to back up on the first save

    second = InMemoryHashIndex()
    second.add(make_fingerprint("file-b", [4, 5]))
    second.save(index_path)

    # The .bak now holds the prior (file-a) snapshot, the primary the new one.
    assert backup.read_text(encoding="utf-8") == first_contents
    assert InMemoryHashIndex.load(backup).file_count == 1
    assert InMemoryHashIndex.load(index_path).file_count == 1
    assert InMemoryHashIndex.load(index_path).search(make_fingerprint("file-b", [4, 5]))[0].file_id == "file-b"


def test_load_raises_when_primary_corrupt_and_no_backup(tmp_path: Path) -> None:
    # A corrupt primary with no .bak must raise (not silently return an empty
    # index, which would then overwrite a good backup on the next save).
    index_path = tmp_path / "index.json"
    index_path.write_text('{"backend": "in_memory", "files": {"file-a"', encoding="utf-8")

    with pytest.raises(ValueError, match="corrupt"):
        InMemoryHashIndex.load(index_path)


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


# ---------------------------------------------------------------------------
# Cross-backend invariant parity (Task 4)
#
# These pin deliberately-implemented invariants whose divergence point had no
# test: the offset tie-break (implemented in three places), the shared base
# scoring/ranking (must stay byte-identical across backends), and the snapshot
# interop preserving postings + metadata + ranks across a save/load between
# different backends. Redis runs via fakeredis (importorskip); Postgres parity
# is gated behind @requires_pg.
# ---------------------------------------------------------------------------


def _parity_backends() -> list[tuple[str, object]]:
    """In-memory, SQLite and (if fakeredis is installed) Redis, fresh each call.

    Postgres is intentionally NOT here: it needs a live server and is gated by
    @requires_pg in its own test below.
    """

    backends: list[tuple[str, object]] = [
        ("in_memory", InMemoryHashIndex()),
        ("sqlite", SQLiteHashIndex(":memory:")),
    ]
    try:
        import fakeredis  # noqa: F401
    except ImportError:
        pass
    else:
        backends.append(("redis", RedisHashIndex(client=_fake_redis(), key_prefix="parity")))
    return backends


def _tie_corpus_and_query() -> tuple[Fingerprint, Fingerprint]:
    """One indexed file with TWO equal-vote offset bins, plus its query.

    The query has four distinct codes all at offset 0. Codes 1,2 sit at offset
    100 in the indexed file (delta 100, two votes); codes 3,4 sit at offset 200
    (delta 200, two votes). The offset histogram is therefore {100: 2, 200: 2}
    -- a genuine tie. The tie-break rule (votes DESC, then offset ASC) must pick
    the SMALLER offset, 100. This is the exact divergence point of the three
    implementations: in-memory ``max(..., key=lambda kv: (kv[1], -kv[0]))`` and
    the SQL ``ROW_NUMBER() ... ORDER BY votes DESC, delta ASC``.
    """

    indexed = fp_with_hashes("tie", [(1, 100), (2, 100), (3, 200), (4, 200)])
    query = fp_with_hashes("q", [(1, 0), (2, 0), (3, 0), (4, 0)])
    return indexed, query


def test_offset_tie_break_picks_smaller_offset_across_backends() -> None:
    # INVARIANT 1: winning-offset tie-break parity. Build a real tie (two offset
    # bins with equal votes) and assert every backend returns the SMALLER offset.
    # If either tie-break flipped (e.g. to offset DESC, or +kv[0]), this fails.
    indexed, query = _tie_corpus_and_query()

    winners: dict[str, tuple[int, int]] = {}
    for name, index in _parity_backends():
        index.add(indexed)
        results = index.search(query, top_k=5)
        assert results, name
        top = results[0]
        # Sanity-check the tie is real on the in-memory backend's histogram:
        # both bins must carry the same (winning) vote count.
        winners[name] = (top.offset, top.aligned_votes)

    # Every backend agrees on the smaller offset (100), with the tied vote count.
    assert set(winners.values()) == {(100, 2)}, winners

    # Belt-and-suspenders: the losing bin really has equal votes, so this
    # exercises the tie-break rather than a plain max.
    histogram: dict[int, int] = {}
    mem = InMemoryHashIndex()
    mem.add(indexed)
    for query_hash in query.hashes:
        for posting in mem.query(query_hash.hash_code):
            delta = posting.time_offset - query_hash.time_offset
            histogram[delta] = histogram.get(delta, 0) + 1
    assert histogram == {100: 2, 200: 2}  # two bins, equal votes -> genuine tie


def test_cross_backend_ranking_is_byte_identical() -> None:
    # INVARIANT 2: shared base scoring/ranking is byte-identical across backends.
    # Index the same corpus into every backend and assert search() returns the
    # SAME (file_id, offset, aligned_votes, score) ordering for one query.
    corpus = [
        fp_with_hashes("alpha", [(1, 10), (2, 20), (3, 30), (4, 40)]),
        fp_with_hashes("beta", [(1, 5), (2, 40), (3, 99), (5, 7)]),
        fp_with_hashes("gamma", [(1, 10), (2, 20), (6, 30)]),
    ]
    query = fp_with_hashes("q", [(1, 0), (2, 10), (3, 20), (4, 30), (5, 1), (6, 2)])

    per_backend: dict[str, list[tuple[str, int, int, float]]] = {}
    for name, index in _parity_backends():
        for fingerprint in corpus:
            index.add(fingerprint)
        per_backend[name] = _search_tuples(index, query, top_k=10)

    # A meaningful ranking (more than one match) actually ordered, not trivial.
    reference = per_backend["in_memory"]
    assert len(reference) >= 3
    assert reference[0][0] == "alpha"  # the coherently-aligned file ranks first
    for name, tuples in per_backend.items():
        assert tuples == reference, (name, tuples, reference)


def test_snapshot_interop_preserves_postings_metadata_and_ranks() -> None:
    # INVARIANT 3: a snapshot save()'d by one backend, load_snapshot'd into
    # another, preserves postings + metadata + ranks identically. Existing tests
    # only check file_count/posting_count + the top result; this pins the full
    # postings set, the metadata map, and the exact ranked ordering.
    import tempfile
    from pathlib import Path as _Path

    corpus = [
        fp_with_hashes("alpha", [(1, 10), (2, 20), (3, 30), (4, 40)]),
        fp_with_hashes("beta", [(1, 5), (2, 40), (3, 99), (5, 7)]),
        fp_with_hashes("gamma", [(1, 10), (2, 20), (6, 30)]),
    ]
    query = fp_with_hashes("q", [(1, 0), (2, 10), (3, 20), (4, 30), (5, 1), (6, 2)])

    def normalized_files(index) -> dict[str, list[tuple[int, int]]]:
        # to_dict() preserves per-file order differently per backend, so compare
        # the postings as sorted multisets -- equality of CONTENT, not order.
        return {
            file_id: sorted((int(code), int(offset)) for code, offset in entries)
            for file_id, entries in index.to_dict()["files"].items()
        }

    def metadata_map(index) -> dict[str, dict]:
        return {file_id: index._metadata_for(file_id) for file_id in index.to_dict()["files"]}

    # Save once from SQLite (a SQL backend, so it round-trips the signed-offset
    # encode), then load into every backend and compare against the source.
    source = SQLiteHashIndex(":memory:")
    for fingerprint in corpus:
        source.add(fingerprint)
    source_files = normalized_files(source)
    source_meta = metadata_map(source)
    source_ranks = _search_tuples(source, query, top_k=10)

    with tempfile.TemporaryDirectory() as directory:
        snapshot = _Path(directory) / "snap.json"
        source.save(snapshot)

        targets: list[tuple[str, object]] = [
            ("in_memory", InMemoryHashIndex().load_snapshot(snapshot)),
            ("sqlite", SQLiteHashIndex(":memory:").load_snapshot(snapshot)),
        ]
        try:
            import fakeredis  # noqa: F401
        except ImportError:
            pass
        else:
            targets.append(
                ("redis", RedisHashIndex(client=_fake_redis(), key_prefix="snap").load_snapshot(snapshot))
            )
        # The classmethod load() entry point must agree with load_snapshot() too.
        targets.append(("in_memory_load", InMemoryHashIndex.load(snapshot)))

        for name, target in targets:
            assert normalized_files(target) == source_files, name
            assert metadata_map(target) == source_meta, name
            assert _search_tuples(target, query, top_k=10) == source_ranks, name


@requires_pg
def test_postgres_offset_tie_break_picks_smaller_offset(pg_index) -> None:
    # INVARIANT 1 for Postgres: its ROW_NUMBER() ... ORDER BY votes DESC, delta
    # ASC must resolve a real tie to the smaller offset, like the others.
    indexed, query = _tie_corpus_and_query()
    pg_index.add(indexed)

    top = pg_index.search(query, top_k=5)[0]
    assert (top.offset, top.aligned_votes) == (100, 2)


@requires_pg
def test_postgres_ranking_matches_in_memory_byte_identical(pg_index) -> None:
    # INVARIANT 2 for Postgres: full (file_id, offset, aligned_votes, score)
    # ordering identical to the in-memory reference.
    corpus = [
        fp_with_hashes("alpha", [(1, 10), (2, 20), (3, 30), (4, 40)]),
        fp_with_hashes("beta", [(1, 5), (2, 40), (3, 99), (5, 7)]),
        fp_with_hashes("gamma", [(1, 10), (2, 20), (6, 30)]),
    ]
    query = fp_with_hashes("q", [(1, 0), (2, 10), (3, 20), (4, 30), (5, 1), (6, 2)])

    mem = InMemoryHashIndex()
    for fingerprint in corpus:
        mem.add(fingerprint)
        pg_index.add(fingerprint)

    assert _search_tuples(pg_index, query, top_k=10) == _search_tuples(mem, query, top_k=10)


@requires_pg
def test_postgres_snapshot_interop_preserves_postings_metadata_and_ranks(pg_index, tmp_path: Path) -> None:
    # INVARIANT 3 for Postgres: a Postgres-written snapshot loads into the
    # in-memory backend with identical postings, metadata, and ranks.
    corpus = [
        fp_with_hashes("alpha", [(1, 10), (2, 20), (3, 30), (4, 40)]),
        fp_with_hashes("beta", [(1, 5), (2, 40), (3, 99), (5, 7)]),
    ]
    query = fp_with_hashes("q", [(1, 0), (2, 10), (3, 20), (4, 30), (5, 1)])
    for fingerprint in corpus:
        pg_index.add(fingerprint)

    def normalized_files(index) -> dict[str, list[tuple[int, int]]]:
        return {
            file_id: sorted((int(code), int(offset)) for code, offset in entries)
            for file_id, entries in index.to_dict()["files"].items()
        }

    snapshot = tmp_path / "snap.json"
    pg_index.save(snapshot)
    loaded = InMemoryHashIndex.load(snapshot)

    assert normalized_files(loaded) == normalized_files(pg_index)
    assert {fid: loaded._metadata_for(fid) for fid in normalized_files(loaded)} == {
        fid: pg_index._metadata_for(fid) for fid in normalized_files(pg_index)
    }
    assert _search_tuples(loaded, query, top_k=10) == _search_tuples(pg_index, query, top_k=10)
