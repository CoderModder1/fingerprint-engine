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
from fingerprint_engine.core.models import (
    Calibration,
    ConstellationHash,
    Fingerprint,
    IndexPosting,
)


def _fake_redis():
    fakeredis = pytest.importorskip("fakeredis", exc_type=ImportError)
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


# ---------------------------------------------------------------------------
# Bulk/transactional ingest parity (Task 3 -- add_many)
#
# add_many() MUST be exactly equivalent to calling add() per fingerprint in
# sequence: same posting_count, same normalized postings + metadata snapshot,
# and same search() rankings -- including the replace case where a file_id is
# re-indexed. These pin that contract for InMemory, SQLite, and (via fakeredis)
# Redis. Postgres uses COPY and is gated behind @requires_pg below.
# ---------------------------------------------------------------------------


def _bulk_parity_factories() -> list[tuple[str, object]]:
    """One factory per available backend: ``factory()`` returns a FRESH index.

    Each parity test needs two independent indexes (sequential vs add_many) and
    several distinct-prefixed Redis namespaces, so a factory (not a single
    pre-built index) keeps them isolated.
    """

    counter = {"n": 0}

    def redis_factory():
        counter["n"] += 1
        return RedisHashIndex(client=_fake_redis(), key_prefix=f"bulk{counter['n']}")

    factories: list[tuple[str, object]] = [
        ("in_memory", InMemoryHashIndex),
        ("sqlite", lambda: SQLiteHashIndex(":memory:")),
    ]
    try:
        import fakeredis  # noqa: F401
    except ImportError:
        pass
    else:
        factories.append(("redis", redis_factory))
    return factories


def _normalized_files(index) -> dict[str, list[tuple[int, int]]]:
    # Compare postings as sorted multisets: to_dict() per-file order differs by
    # backend, but the CONTENT must be identical.
    return {
        file_id: sorted((int(code), int(offset)) for code, offset in entries)
        for file_id, entries in index.to_dict()["files"].items()
    }


def _metadata_map(index) -> dict[str, dict]:
    return {file_id: index._metadata_for(file_id) for file_id in index.to_dict()["files"]}


def _assert_bulk_equivalent(name, sequential, bulk, queries) -> None:
    # The core parity gate: add_many() == sequential add() for every observable.
    assert bulk.file_count == sequential.file_count, name
    assert bulk.posting_count == sequential.posting_count, name
    assert _normalized_files(bulk) == _normalized_files(sequential), name
    assert _metadata_map(bulk) == _metadata_map(sequential), name
    for query in queries:
        assert _search_tuples(bulk, query, top_k=20) == _search_tuples(sequential, query, top_k=20), (
            name, query.file_id,
        )


def _bulk_corpus() -> list[Fingerprint]:
    return [
        fp_with_hashes("alpha", [(1, 10), (2, 20), (3, 30), (4, 40)]),
        fp_with_hashes("beta", [(1, 5), (2, 40), (3, 99), (5, 7)]),
        fp_with_hashes("gamma", [(1, 10), (2, 20), (6, 30)]),
        # A file with zero hashes: add()/remove() must still track membership.
        fp_with_hashes("empty", []),
    ]


def _bulk_queries() -> list[Fingerprint]:
    return [
        fp_with_hashes("q1", [(1, 0), (2, 10), (3, 20), (4, 30), (5, 1), (6, 2)]),
        fp_with_hashes("q2", [(1, 0), (2, 0), (3, 0)]),
    ]


def test_add_many_equals_sequential_add_across_backends() -> None:
    corpus = _bulk_corpus()
    queries = _bulk_queries()
    for name, factory in _bulk_parity_factories():
        sequential = factory()
        for fingerprint in corpus:
            sequential.add(fingerprint)
        bulk = factory()
        bulk.add_many(corpus)
        _assert_bulk_equivalent(name, sequential, bulk, queries)


def test_add_many_preserves_replace_semantics_across_backends() -> None:
    # A batch containing a file_id already present (and a duplicate WITHIN the
    # batch) must end in the same state as sequential add(): last writer wins,
    # pre-existing postings removed exactly once.
    queries = _bulk_queries()
    # Re-index list: alpha replaced (different postings), beta new, alpha AGAIN
    # later in the same batch (intra-batch duplicate -> the final alpha wins).
    reindex = [
        fp_with_hashes("alpha", [(7, 1), (8, 2)]),
        fp_with_hashes("beta", [(1, 5), (2, 40)]),
        fp_with_hashes("alpha", [(9, 3), (10, 4), (11, 5)]),  # later duplicate wins
    ]
    for name, factory in _bulk_parity_factories():
        sequential = factory()
        # Pre-populate so add_many must REPLACE an existing alpha.
        for fingerprint in [fp_with_hashes("alpha", [(1, 100), (2, 200)]), fp_with_hashes("gamma", [(3, 1)])]:
            sequential.add(fingerprint)
        bulk = factory()
        for fingerprint in [fp_with_hashes("alpha", [(1, 100), (2, 200)]), fp_with_hashes("gamma", [(3, 1)])]:
            bulk.add(fingerprint)

        for fingerprint in reindex:
            sequential.add(fingerprint)  # sequential reference path
        bulk.add_many(reindex)           # bulk path under test

        _assert_bulk_equivalent(name, sequential, bulk, queries)
        # Concrete replace check: the later alpha (codes 9,10,11) is the survivor,
        # the pre-batch alpha (codes 1,2) is gone -- on both paths.
        assert _normalized_files(bulk)["alpha"] == [(9, 3), (10, 4), (11, 5)], name


def test_add_many_empty_iterable_is_noop_across_backends() -> None:
    for name, factory in _bulk_parity_factories():
        index = factory()
        index.add_many([])
        assert index.file_count == 0, name
        assert index.posting_count == 0, name


def test_sqlite_add_many_uses_single_commit() -> None:
    # The whole point of the SQLite override: ONE commit for the batch, not one
    # per file. We prove the commit-count drop by counting sqlite3 commits via a
    # wrapped connection, and (separately) that synchronous=NORMAL is set.
    import sqlite3 as _sqlite3

    commits = {"n": 0}
    real_connect = _sqlite3.connect

    class CountingConnection:
        def __init__(self, conn):
            self._conn = conn

        def commit(self):
            commits["n"] += 1
            return self._conn.commit()

        def __getattr__(self, attr):
            return getattr(self._conn, attr)

    raw = real_connect(":memory:", check_same_thread=False)
    index = SQLiteHashIndex(connection=CountingConnection(raw))

    # synchronous=NORMAL (value 1) is set by _init_schema for the WAL pairing.
    assert int(raw.execute("PRAGMA synchronous").fetchone()[0]) == 1

    corpus = [make_fingerprint(f"f{i}", [1, 2, 3, 4]) for i in range(25)]
    commits["n"] = 0  # ignore schema-setup commits; count only the ingest
    index.add_many(corpus)

    # add() would commit at least once per file (>=25); add_many commits ONCE.
    assert commits["n"] == 1
    assert index.file_count == 25
    assert index.posting_count == 100


@requires_pg
def test_pg_add_many_equals_sequential_add(pg_index) -> None:
    # Postgres parity: COPY-based add_many() must match sequential add(). Uses a
    # second prefix for the sequential reference so the two never share tables.
    corpus = _bulk_corpus()
    queries = _bulk_queries()

    sequential = PostgresHashIndex(dsn=PG_DSN, table_prefix="fp_pytest_seq")
    with sequential._conn.cursor() as cur:
        cur.execute(f"TRUNCATE {sequential._files_table}, {sequential._postings_table}")
    sequential._conn.commit()
    try:
        for fingerprint in corpus:
            sequential.add(fingerprint)
        pg_index.add_many(corpus)
        _assert_bulk_equivalent("postgres", sequential, pg_index, queries)
    finally:
        with sequential._conn.cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS {sequential._postings_table}, {sequential._files_table}")
        sequential._conn.commit()
        sequential.close()


# ---------------------------------------------------------------------------
# Enumeration API parity (Task 1 -- list_files / iter_metadata / contains)
#
# list_files() and iter_metadata() MUST be byte-identical across every backend
# and reflect adds AND removes; iter_metadata() MUST yield exactly what
# _metadata_for() returns, in list_files() order, without going through the
# heavy to_dict(). contains()/__contains__ MUST agree with list_files()
# membership. Redis runs via fakeredis; Postgres parity is gated by @requires_pg.
# ---------------------------------------------------------------------------


def test_list_files_and_iter_metadata_parity_across_backends() -> None:
    corpus = _bulk_corpus()  # includes an "empty" (zero-hash) file: still listed
    expected_ids = sorted(fingerprint.file_id for fingerprint in corpus)

    per_backend_ids: dict[str, list[str]] = {}
    per_backend_meta: dict[str, list[dict]] = {}
    for name, index in _parity_backends():
        for fingerprint in corpus:
            index.add(fingerprint)

        listed = index.list_files()
        # Sorted, deterministic, and covers every added file (empty one included).
        assert listed == expected_ids, name
        per_backend_ids[name] = listed

        # iter_metadata() must yield exactly _metadata_for() for each id, IN ORDER,
        # and not depend on to_dict() (it streams list_files()+_metadata_for()).
        streamed = list(index.iter_metadata())
        assert [m["file_id"] for m in streamed] == expected_ids, name
        assert streamed == [index._metadata_for(fid) for fid in listed], name
        per_backend_meta[name] = streamed

        # Membership: contains()/__contains__ agree with list_files().
        for fid in expected_ids:
            assert index.contains(fid), (name, fid)
            assert fid in index, (name, fid)
        assert not index.contains("does-not-exist"), name
        assert "does-not-exist" not in index, name
        assert 12345 not in index, name  # non-str membership is False, not a crash

    # Cross-backend byte-identical enumeration (ids and full metadata dicts).
    reference_ids = per_backend_ids["in_memory"]
    reference_meta = per_backend_meta["in_memory"]
    for name in per_backend_ids:
        assert per_backend_ids[name] == reference_ids, name
        assert per_backend_meta[name] == reference_meta, name


def test_list_files_reflects_adds_and_removes_across_backends() -> None:
    for name, index in _parity_backends():
        assert index.list_files() == [], name  # empty index enumerates to nothing
        assert list(index.iter_metadata()) == [], name

        index.add(fp_with_hashes("alpha", [(1, 10), (2, 20)]))
        index.add(fp_with_hashes("beta", [(3, 5)]))
        assert index.list_files() == ["alpha", "beta"], name
        assert "alpha" in index and "beta" in index, name

        index.remove("alpha")
        assert index.list_files() == ["beta"], name
        assert "alpha" not in index, name
        assert [m["file_id"] for m in index.iter_metadata()] == ["beta"], name

        # Re-adding (replace) a file_id keeps it listed exactly once.
        index.add(fp_with_hashes("beta", [(9, 1), (9, 2)]))
        assert index.list_files() == ["beta"], name
        assert index._metadata_for("beta")["hash_count"] == 2, name


def test_iter_metadata_does_not_call_to_dict() -> None:
    # iter_metadata() must stream list_files()+_metadata_for(), NOT build the heavy
    # whole-index to_dict(). Trip a flag if to_dict() is touched during iteration.
    index = InMemoryHashIndex()
    for fingerprint in _bulk_corpus():
        index.add(fingerprint)

    called = {"to_dict": False}
    original = index.to_dict

    def _tripwire() -> dict:
        called["to_dict"] = True
        return original()

    index.to_dict = _tripwire  # type: ignore[method-assign]
    streamed = list(index.iter_metadata())

    assert not called["to_dict"]
    assert [m["file_id"] for m in streamed] == sorted(fp.file_id for fp in _bulk_corpus())


def test_list_files_and_iter_metadata_survive_snapshot_round_trip(tmp_path: Path) -> None:
    # Enumeration must reflect a load_snapshot() bulk import, not just live adds.
    corpus = _bulk_corpus()
    source = SQLiteHashIndex(":memory:")
    for fingerprint in corpus:
        source.add(fingerprint)
    snapshot = tmp_path / "snap.json"
    source.save(snapshot)

    loaded = InMemoryHashIndex.load(snapshot)
    assert loaded.list_files() == source.list_files()
    assert list(loaded.iter_metadata()) == list(source.iter_metadata())


@requires_pg
def test_pg_list_files_and_iter_metadata_match_in_memory(pg_index) -> None:
    # Enumeration parity for Postgres against the in-memory reference, including
    # the empty (zero-hash) file and removal.
    corpus = _bulk_corpus()
    mem = InMemoryHashIndex()
    for fingerprint in corpus:
        mem.add(fingerprint)
        pg_index.add(fingerprint)

    assert pg_index.list_files() == mem.list_files()
    assert list(pg_index.iter_metadata()) == list(mem.iter_metadata())
    for fid in mem.list_files():
        assert pg_index.contains(fid)
        assert fid in pg_index
    assert "missing" not in pg_index

    pg_index.remove("alpha")
    mem.remove("alpha")
    assert pg_index.list_files() == mem.list_files()
    assert "alpha" not in pg_index


@requires_pg
def test_pg_add_many_preserves_replace_semantics(pg_index) -> None:
    queries = _bulk_queries()
    pg_index.add(fp_with_hashes("alpha", [(1, 100), (2, 200)]))
    pg_index.add(fp_with_hashes("gamma", [(3, 1)]))

    reference = PostgresHashIndex(dsn=PG_DSN, table_prefix="fp_pytest_seq")
    with reference._conn.cursor() as cur:
        cur.execute(f"TRUNCATE {reference._files_table}, {reference._postings_table}")
    reference._conn.commit()
    try:
        reference.add(fp_with_hashes("alpha", [(1, 100), (2, 200)]))
        reference.add(fp_with_hashes("gamma", [(3, 1)]))
        reindex = [
            fp_with_hashes("alpha", [(7, 1), (8, 2)]),
            fp_with_hashes("beta", [(1, 5), (2, 40)]),
            fp_with_hashes("alpha", [(9, 3), (10, 4), (11, 5)]),
        ]
        for fingerprint in reindex:
            reference.add(fingerprint)
        pg_index.add_many(reindex)
        _assert_bulk_equivalent("postgres", reference, pg_index, queries)
        assert _normalized_files(pg_index)["alpha"] == [(9, 3), (10, 4), (11, 5)]
    finally:
        with reference._conn.cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS {reference._postings_table}, {reference._files_table}")
        reference._conn.commit()
        reference.close()


# ---------------------------------------------------------------------------
# OPT-IN offset-tolerant voting (item 2 -- multi-edit near-dups)
#
# search(offset_tolerance>0) sums the winning bin's votes over +-tolerance
# adjacent delta bins so multi-edit near-dups whose votes fragment across
# adjacent deltas recover their score. The default (0 / unset) MUST stay
# byte-identical to exact-bin voting, and any tolerance MUST band identically
# across every backend (the SQL backends band the server-side histogram in the
# shared Python reducer, the in-memory backend bands its own histogram).
# ---------------------------------------------------------------------------


def _fragmented_corpus_and_query() -> tuple[list[Fingerprint], Fingerprint]:
    """A multi-edit query whose true match fragments across THREE delta bins.

    ``doc`` is codes 1..12 at consecutive offsets. The query reproduces all 12
    codes but in three 4-code fragments shifted by 0, +1, +2 frames (as three
    insertions would shift the absolute frame index of everything after them),
    so the true offset histogram for ``doc`` is {0: 4, 1: 4, 2: 4} -- max single
    bin 4. ``decoy`` shares six codes that all align in ONE bin (6 votes), so it
    beats the fragmented true match at exact-bin (tolerance 0) voting. A
    tolerance that spans the three fragments recombines them to 12 and recovers
    ``doc`` as the rank-1 match.
    """

    doc = fp_with_hashes("doc", [(code, code - 1) for code in range(1, 13)])
    decoy = fp_with_hashes("decoy", [(100 + i, i) for i in range(6)])
    pairs = (
        [(code, code - 1) for code in range(1, 5)]        # fragment A: delta 0
        + [(code, (code - 1) - 1) for code in range(5, 9)]   # fragment B: delta +1
        + [(code, (code - 1) - 2) for code in range(9, 13)]  # fragment C: delta +2
        + [(100 + i, i) for i in range(6)]                   # decoy codes, delta 0
    )
    return [doc, decoy], fp_with_hashes("q", pairs)


def test_offset_tolerance_default_is_byte_identical_across_backends() -> None:
    # DEFAULT-PRESERVING: an unset / explicit-0 offset_tolerance, and a default
    # Calibration (offset_tolerance 0), must all reproduce the EXACT same ranked
    # (file_id, offset, aligned_votes, score) tuples as the legacy exact-bin
    # search -- on a corpus that has a genuine multi-bin histogram.
    corpus, query = _fragmented_corpus_and_query()
    for name, index in _parity_backends():
        for fingerprint in corpus:
            index.add(fingerprint)
        baseline = _search_tuples(index, query, top_k=10)
        # The fragmented true match loses to the single-bin decoy at exact bins.
        assert baseline[0][0] == "decoy", name
        # Every spelling of "off" yields the identical ranking.
        assert _search_tuples(index, query, top_k=10) == baseline, name
        assert [
            (r.file_id, r.offset, r.aligned_votes, r.score)
            for r in index.search(query, top_k=10, offset_tolerance=0)
        ] == baseline, name
        assert [
            (r.file_id, r.offset, r.aligned_votes, r.score)
            for r in index.search(query, top_k=10, calibration=Calibration())
        ] == baseline, name
        assert [
            (r.file_id, r.offset, r.aligned_votes, r.score)
            for r in index.search(query, top_k=10, calibration=Calibration(offset_tolerance=0))
        ] == baseline, name


def test_offset_tolerance_recovers_multi_edit_and_is_parity_identical() -> None:
    # tolerance>0 recovers the fragmented true match AND every backend bands
    # identically (full ranked-tuple parity), at tolerance 1 and 2.
    corpus, query = _fragmented_corpus_and_query()
    per_tolerance: dict[int, dict[str, list[tuple[str, int, int, float]]]] = {1: {}, 2: {}}
    for name, index in _parity_backends():
        for fingerprint in corpus:
            index.add(fingerprint)
        # tolerance 0 fails to rank the true match first...
        assert index.search(query, top_k=1, offset_tolerance=0)[0].file_id == "decoy", name
        for tolerance in (1, 2):
            results = index.search(query, top_k=10, offset_tolerance=tolerance)
            top = results[0]
            # ...tolerance recombines the three 4-vote fragments into 12 votes.
            assert top.file_id == "doc", (name, tolerance)
            assert top.aligned_votes == 12, (name, tolerance)
            assert top.confidence == 1.0, (name, tolerance)
            per_tolerance[tolerance][name] = [
                (r.file_id, r.offset, r.aligned_votes, r.score) for r in results
            ]

    # Cross-backend parity: every backend produced the identical banded ranking.
    for tolerance, by_backend in per_tolerance.items():
        reference = by_backend["in_memory"]
        for name, tuples in by_backend.items():
            assert tuples == reference, (tolerance, name, tuples, reference)
    # Documented band-centre behaviour: tol=1's winning centre is delta 1 (its
    # band [0, 2] covers all three fragments); tol=2 picks the SMALLER centre 0
    # (its band [-2, 2] also covers them) per the votes-DESC, delta-ASC rule.
    assert per_tolerance[1]["in_memory"][0][1] == 1
    assert per_tolerance[2]["in_memory"][0][1] == 0


def test_offset_tolerance_via_calibration_field_and_explicit_arg_precedence() -> None:
    # The Calibration.offset_tolerance field drives banding when no explicit arg
    # is given; an explicit search(offset_tolerance=...) overrides the field.
    corpus, query = _fragmented_corpus_and_query()
    index = InMemoryHashIndex()
    for fingerprint in corpus:
        index.add(fingerprint)

    # Field alone (no explicit arg) bands and recovers the true match.
    via_field = index.search(query, top_k=1, calibration=Calibration(offset_tolerance=1))
    assert via_field[0].file_id == "doc"

    # Explicit 0 overrides a banding calibration -> back to exact-bin (decoy).
    overridden = index.search(
        query, top_k=1, calibration=Calibration(offset_tolerance=2), offset_tolerance=0
    )
    assert overridden[0].file_id == "decoy"

    # Explicit tolerance overrides a zero-tolerance calibration -> bands.
    forced = index.search(
        query, top_k=1, calibration=Calibration(offset_tolerance=0), offset_tolerance=1
    )
    assert forced[0].file_id == "doc"


def test_offset_tolerance_negative_is_rejected() -> None:
    corpus, query = _fragmented_corpus_and_query()
    index = InMemoryHashIndex()
    for fingerprint in corpus:
        index.add(fingerprint)
    with pytest.raises(ValueError, match="offset_tolerance"):
        index.search(query, offset_tolerance=-1)
    with pytest.raises(ValueError, match="offset_tolerance"):
        Calibration(offset_tolerance=-1)


def test_banded_winner_matches_exact_bin_when_tolerance_zero() -> None:
    # Unit-level: with tolerance 0 the banded winner equals the legacy
    # max(items, key=(votes, -delta)) for a histogram with a genuine tie.
    histogram = {100: 2, 200: 2, 50: 1}
    assert InMemoryHashIndex._banded_winner(histogram, 0) == (100, 2)  # votes tie -> smaller delta
    # A clear single winner is returned exactly.
    assert InMemoryHashIndex._banded_winner({5: 1, 6: 3, 7: 1}, 0) == (6, 3)


def test_banded_winner_sums_adjacent_bins_and_breaks_ties_by_smaller_centre() -> None:
    # Two fragments at deltas 10 and 11 (3 votes each) plus an isolated 6-vote
    # bin at delta 40. At tolerance 0 the 40-bin wins (6 > 3). At tolerance 1 the
    # band around centre 10 (or 11) sums to 6, tying the 40-bin; the smaller
    # centre (10) wins the tie.
    histogram = {10: 3, 11: 3, 40: 6}
    assert InMemoryHashIndex._banded_winner(histogram, 0) == (40, 6)
    assert InMemoryHashIndex._banded_winner(histogram, 1) == (10, 6)
    # A wider band that also pulls in the 40-bin only at a far centre still
    # prefers the densest, smallest-centre window.
    assert InMemoryHashIndex._banded_winner({10: 3, 11: 3, 12: 3}, 1) == (11, 9)


def test_offset_tolerance_does_not_change_a_clean_self_match() -> None:
    # A coherent (single-bin) match is unaffected by banding: a self-search still
    # aligns all votes in one bin, so the offset / aligned_votes / score are the
    # SAME at tolerance 0, 1 and 2 (banding only ADDS neighbouring bins, of which
    # there are none here). Guards against banding perturbing the common case.
    index = InMemoryHashIndex()
    index.add(make_fingerprint("aligned", [10, 20, 30, 40]))
    index.add(make_fingerprint("scattered", [2, 40, 99, 125]))
    query = make_fingerprint("query", [3, 13, 23, 33])

    exact = _search_tuples(index, query, top_k=5)
    assert exact[0][0] == "aligned" and exact[0][2] == 4
    for tolerance in (1, 2):
        assert _search_tuples(index, query, top_k=5) == exact  # unset default
        banded = [
            (r.file_id, r.offset, r.aligned_votes, r.score)
            for r in index.search(query, top_k=5, offset_tolerance=tolerance)
        ]
        # The coherent winner is unchanged; "scattered" may gain banded votes but
        # must never overtake the clean full-confidence match.
        assert banded[0] == exact[0], tolerance


# ---------------------------------------------------------------------------
# OPT-IN ANN/LSH-style candidate prefilter (candidate_limit). Default None =
# OFF = full exact search, byte-identical. A generous limit (>= top_k, and a
# superset of the true matches) must leave the ranking identical.
# ---------------------------------------------------------------------------


def _prefilter_corpus() -> list[Fingerprint]:
    """A corpus with a clear shared-hash gradient between files.

    Query uses codes 1..6 at offset 0. "alpha" shares all six coherently
    (strongest, highest shared-hash count); "beta" shares five; "gamma" three;
    "delta" two; "noise" shares none of the query codes (uses codes 90..92), so
    it is never a candidate. This makes the shared-hash ranking unambiguous so a
    prefilter cut at a known boundary is deterministic.
    """

    return [
        fp_with_hashes("alpha", [(1, 10), (2, 20), (3, 30), (4, 40), (5, 50), (6, 60)]),
        fp_with_hashes("beta", [(1, 5), (2, 6), (3, 7), (4, 8), (5, 9)]),
        fp_with_hashes("gamma", [(1, 100), (2, 7), (3, 30)]),
        fp_with_hashes("delta", [(1, 3), (2, 99)]),
        fp_with_hashes("noise", [(90, 1), (91, 2), (92, 3)]),
    ]


def _prefilter_query() -> Fingerprint:
    return fp_with_hashes("q", [(1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0)])


def test_candidate_limit_none_is_byte_identical_full_search_across_backends() -> None:
    # DEFAULT-PRESERVING: candidate_limit=None (the default) must be the full
    # exact search -- byte-identical to omitting the argument entirely -- on
    # every backend.
    corpus = _prefilter_corpus()
    query = _prefilter_query()
    for name, index in _parity_backends():
        for fingerprint in corpus:
            index.add(fingerprint)
        full = _search_tuples(index, query, top_k=10)
        explicit_none = [
            (r.file_id, r.offset, r.aligned_votes, r.score)
            for r in index.search(query, top_k=10, candidate_limit=None)
        ]
        assert explicit_none == full, name
        # Sanity: every shared-hash file ranked, "noise" (no shared code) absent.
        ranked_ids = {row[0] for row in full}
        assert "alpha" in ranked_ids and "noise" not in ranked_ids, name


def test_candidate_limit_superset_keeps_ranking_identical_across_backends() -> None:
    # When the candidate set is a SUPERSET of the true top-k, the final ranking
    # is identical to full search. There are four shared-hash files; a limit of
    # 4 (or more) retains them all, so the ranking must match the full search
    # byte-for-byte on every backend.
    corpus = _prefilter_corpus()
    query = _prefilter_query()
    for name, index in _parity_backends():
        for fingerprint in corpus:
            index.add(fingerprint)
        full = _search_tuples(index, query, top_k=10)
        for limit in (4, 5, 10, 100):
            limited = [
                (r.file_id, r.offset, r.aligned_votes, r.score)
                for r in index.search(query, top_k=10, candidate_limit=limit)
            ]
            assert limited == full, (name, limit)


def test_candidate_limit_keeps_self_match_recall_for_high_overlap() -> None:
    # A self/near-dup query has maximal shared-hash count, so even a TIGHT limit
    # (1) keeps the true top-1. recall@1 stays 1.0 for the strongest match.
    corpus = _prefilter_corpus()
    query = _prefilter_query()
    for name, index in _parity_backends():
        for fingerprint in corpus:
            index.add(fingerprint)
        top1 = index.search(query, top_k=1, candidate_limit=1)
        assert top1 and top1[0].file_id == "alpha", name


def test_candidate_limit_reduces_candidate_set_below_corpus() -> None:
    # The speed proxy: a tight limit aggregates strictly fewer files than the
    # full search would. With four shared-hash files, limit=2 yields exactly two
    # results (the two strongest), proving the candidate set is smaller than the
    # set the full search scores.
    corpus = _prefilter_corpus()
    query = _prefilter_query()
    for name, index in _parity_backends():
        for fingerprint in corpus:
            index.add(fingerprint)
        full = index.search(query, top_k=10)
        limited = index.search(query, top_k=10, candidate_limit=2)
        assert len(full) == 4, name  # alpha, beta, gamma, delta all share a code
        assert len(limited) == 2, name  # only the top-2 shared-hash files scored
        assert [r.file_id for r in limited] == ["alpha", "beta"], name


def test_candidate_limit_zero_returns_no_results_across_backends() -> None:
    corpus = _prefilter_corpus()
    query = _prefilter_query()
    for name, index in _parity_backends():
        for fingerprint in corpus:
            index.add(fingerprint)
        assert index.search(query, top_k=10, candidate_limit=0) == [], name


def test_candidate_limit_negative_is_rejected() -> None:
    index = InMemoryHashIndex()
    index.add(fp_with_hashes("alpha", [(1, 10), (2, 20)]))
    query = fp_with_hashes("q", [(1, 0), (2, 0)])
    with pytest.raises(ValueError, match="candidate_limit"):
        index.search(query, top_k=5, candidate_limit=-1)


def test_candidate_prefilter_recall_on_harness_corpus() -> None:
    # MEASUREMENT-as-test: on the deterministic accuracy harness corpus, a
    # generous candidate_limit keeps exact self-match recall@1 at 1.0 and the
    # candidate set strictly smaller than the corpus for a self-query.
    import tempfile

    import numpy as np

    from benchmarks.accuracy import _write_text_corpus
    from fingerprint_engine.core.fingerprinter import Fingerprinter

    fingerprinter = Fingerprinter()
    rng = np.random.default_rng(1234)
    with tempfile.TemporaryDirectory(prefix="fp_prefilter_") as tmp:
        paths, _texts = _write_text_corpus(rng, 36, Path(tmp))
        fingerprints = [fingerprinter.fingerprint_file(p) for p in paths]
    index = InMemoryHashIndex()
    index.add_many(fingerprints)

    corpus_size = len(fingerprints)
    hits_full = 0
    hits_pref = 0
    candidate_sizes: list[int] = []
    for fingerprint in fingerprints:
        full = index.search(fingerprint, top_k=1)
        pref = index.search(fingerprint, top_k=1, candidate_limit=10)
        if full and full[0].file_id == fingerprint.file_id:
            hits_full += 1
        if pref and pref[0].file_id == fingerprint.file_id:
            hits_pref += 1
        candidate_sizes.append(len(index._select_candidates(fingerprint, 10)))  # type: ignore[arg-type]

    # Full and prefiltered exact recall@1 are both perfect for self-matches.
    assert hits_full == corpus_size
    assert hits_pref == corpus_size
    # The prefilter caps the candidate set well below the corpus size (the speed
    # proxy): a self-query never expands the candidate set beyond the limit.
    assert max(candidate_sizes) <= 10 < corpus_size


# --------------------------------------------------------------------------- #
# Integer file_id surrogate key (internal storage optimization).
#
# InMemoryHashIndex stores a compact integer surrogate per posting instead of
# the 64-char SHA-256 file_id, mapping back to the str file_id only at the
# query/aggregate boundary. These tests pin that the optimization is fully
# OUTPUT-PRESERVING: search rankings, to_dict snapshots, list_files,
# iter_metadata, query, and query_many are byte-identical to storing the str
# verbatim, plus the internal surrogate invariants (retire-not-reuse) hold.
# --------------------------------------------------------------------------- #

# Realistic 64-hex-char SHA-256-shaped file_ids (the value the surrogate replaces).
_SURROGATE_CORPUS = [
    ("a" * 64, [(1000, 10), (1001, 20), (1002, 30), (1003, 40)]),
    ("b" * 64, [(1000, 2), (1001, 40), (1004, 99), (1005, 125)]),
    ("c" * 64, [(1001, 21), (1002, 31), (1006, 7)]),
    ("d" * 64, [(1000, 11), (1001, 22), (1002, 33), (1003, 44), (1007, 88)]),
]


def _build_surrogate_index() -> InMemoryHashIndex:
    index = InMemoryHashIndex()
    for file_id, code_offsets in _SURROGATE_CORPUS:
        index.add(fp_with_hashes(file_id, code_offsets))
    return index


def _reference_postings() -> dict[int, list[tuple[str, int]]]:
    """The (file_id, time_offset) posting lists a str-storing index would hold.

    Recomputed straight from the corpus definition, in add() insertion order,
    so it is an independent oracle for what query()/query_many() must return.
    """

    expected: dict[int, list[tuple[str, int]]] = {}
    for file_id, code_offsets in _SURROGATE_CORPUS:
        for code, offset in code_offsets:
            expected.setdefault(code, []).append((file_id, offset))
    return expected


def test_surrogate_query_returns_str_file_ids_in_stored_order() -> None:
    # query() maps each surrogate back to the original str file_id, preserving
    # the public IndexPosting return type/values and the stored posting order.
    index = _build_surrogate_index()
    expected = _reference_postings()
    for code, expected_postings in expected.items():
        got = index.query(code)
        assert [(p.file_id, p.time_offset) for p in got] == expected_postings
        assert all(p.hash_code == code for p in got)
        assert all(isinstance(p.file_id, str) and len(p.file_id) == 64 for p in got)


def test_surrogate_query_many_matches_per_code_query() -> None:
    # query_many is parity-identical to per-code query() (the base contract),
    # which must survive the surrogate mapping unchanged.
    index = _build_surrogate_index()
    codes = [code for code, _ in _reference_postings().items()]
    batched = index.query_many(codes + [999999])  # include a miss
    for code in codes:
        assert sorted((p.file_id, p.time_offset) for p in batched[code]) == \
               sorted((p.file_id, p.time_offset) for p in index.query(code))
    assert batched[999999] == []


def test_surrogate_postings_store_no_file_id_strings() -> None:
    # The actual footprint win: every stored posting is a compact (int, int)
    # pair -- no 64-char file_id string is held per posting.
    index = _build_surrogate_index()
    for postings in index._postings.values():
        for posting in postings:
            assert isinstance(posting, tuple) and len(posting) == 2
            assert isinstance(posting[0], int)   # surrogate, not a str file_id
            assert isinstance(posting[1], int)   # time_offset
    # Every live surrogate resolves back to a real file_id and round-trips.
    assert set(index._id_to_fid.values()) == {fid for fid, _ in _SURROGATE_CORPUS}
    assert index._fid_to_id == {fid: sid for sid, fid in index._id_to_fid.items()}


def test_surrogate_search_to_dict_list_files_iter_metadata_identical() -> None:
    # OUTPUT-PRESERVING PROOF: every observable matches a reference index built
    # from the SAME fingerprints. (Both use the surrogate storage, but the
    # reference is reconstructed via the public to_dict/from_dict round-trip,
    # exercising the full encode/decode path independent of the live add path.)
    index = _build_surrogate_index()
    reference = InMemoryHashIndex.from_dict(index.to_dict())

    query = fp_with_hashes("q" * 64, [(1000, 0), (1001, 0), (1002, 0), (1003, 0)])
    assert _search_tuples(index, query, top_k=10) == _search_tuples(reference, query, top_k=10)
    # Full SearchResult equality (score, votes, offset, confidence, metadata).
    assert index.search(query, top_k=10) == reference.search(query, top_k=10)

    assert index.to_dict() == reference.to_dict()
    assert index.list_files() == reference.list_files()
    assert list(index.iter_metadata()) == list(reference.iter_metadata())
    assert [index._metadata_for(fid) for fid in index.list_files()] == \
           [reference._metadata_for(fid) for fid in reference.list_files()]
    assert index.file_count == reference.file_count
    assert index.posting_count == reference.posting_count


def test_surrogate_to_dict_keys_are_str_file_ids() -> None:
    # The snapshot is keyed by the original 64-char file_id (NOT the surrogate),
    # so snapshots stay portable and cross-backend identical.
    index = _build_surrogate_index()
    snapshot = index.to_dict()
    assert set(snapshot["files"]) == {fid for fid, _ in _SURROGATE_CORPUS}
    assert set(snapshot["metadata"]) == {fid for fid, _ in _SURROGATE_CORPUS}
    # The corpus's first file: stored entries are (hash_code, time_offset) pairs
    # with the real hash codes, never surrogate ints. (InMemory.to_dict returns
    # _file_entries verbatim as tuples -- pre-existing behaviour, unchanged by
    # the surrogate work; the JSON snapshot round-trip normalizes them to lists.)
    first_id, first_pairs = _SURROGATE_CORPUS[0]
    assert snapshot["files"][first_id] == [(c, o) for c, o in first_pairs]


def test_surrogate_retained_on_remove_and_reused_on_readd() -> None:
    # remove() deliberately KEEPS the surrogate mappings alive (so a lock-free
    # reader can always resolve a just-removed posting's surrogate instead of
    # racing into a KeyError), and a re-added file_id REUSES that same surrogate
    # (re-indexing never grows the maps). The retained surrogate is postings-less,
    # so it is invisible to every query/search.
    index = InMemoryHashIndex()
    index.add(fp_with_hashes("x" * 64, [(1, 10), (2, 20)]))
    first_surrogate = index._fid_to_id["x" * 64]
    index.remove("x" * 64)
    # Mappings retained across remove() -- still resolvable, never KeyError.
    assert index._fid_to_id["x" * 64] == first_surrogate
    assert index._id_to_fid[first_surrogate] == "x" * 64
    assert index.query(1) == []  # postings gone (the observable effect)

    index.add(fp_with_hashes("x" * 64, [(1, 10), (2, 20)]))
    second_surrogate = index._fid_to_id["x" * 64]
    assert second_surrogate == first_surrogate  # reused, not a fresh allocation
    # query() still resolves to the right str file_id after the surrogate churn.
    assert [(p.file_id, p.time_offset) for p in index.query(1)] == [("x" * 64, 10)]


def test_surrogate_replace_keeps_output_correct() -> None:
    # add() of an existing file_id replaces it: old postings are gone, the new
    # ones resolve correctly, counts are exact, and the file's surrogate is
    # REUSED (replace re-interns the same id -> same surrogate, maps don't grow).
    index = InMemoryHashIndex()
    index.add(fp_with_hashes("a" * 64, [(1, 1), (2, 2), (3, 3)]))
    index.add(fp_with_hashes("b" * 64, [(1, 5)]))
    old_a = index._fid_to_id["a" * 64]
    index.add(fp_with_hashes("a" * 64, [(9, 99)]))  # replace a
    new_a = index._fid_to_id["a" * 64]
    assert new_a == old_a  # reused across replace
    assert index._id_to_fid[old_a] == "a" * 64  # mapping still valid
    assert index.file_count == 2
    assert index.posting_count == 2  # a:1 + b:1
    assert index.query(1) == [IndexPosting(file_id="b" * 64, hash_code=1, time_offset=5)]
    assert index.query(9) == [IndexPosting(file_id="a" * 64, hash_code=9, time_offset=99)]
    assert index.query(2) == []  # old a posting removed


def test_surrogate_prune_stop_hashes_output_identical() -> None:
    # prune_stop_hashes uses distinct surrogates as the document-frequency
    # measure; the result (removed count, remaining postings, recalibrated
    # hash_count, search) matches a from_dict reconstruction of the same state.
    index = InMemoryHashIndex()
    # code 1000 appears in all 4 files (df=1.0 -> stop at ratio 0.5); others rare.
    for i in range(4):
        fid = chr(ord("a") + i) * 64
        index.add(fp_with_hashes(fid, [(1000, i), (2000 + i, i * 2)]))
    removed = index.prune_stop_hashes(max_df_ratio=0.5)
    assert removed == 4  # the 4 postings of code 1000
    assert index.query(1000) == []
    # Each file keeps exactly its one rare posting; hash_count recalibrated.
    for i in range(4):
        fid = chr(ord("a") + i) * 64
        assert index._metadata_for(fid)["hash_count"] == 1
        assert [(p.file_id, p.time_offset) for p in index.query(2000 + i)] == [(fid, i * 2)]
    # A from_dict reconstruction of the pruned snapshot is output-identical.
    reference = InMemoryHashIndex.from_dict(index.to_dict())
    assert index.to_dict() == reference.to_dict()
    assert list(index.iter_metadata()) == list(reference.iter_metadata())


def test_surrogate_cross_backend_parity_with_sqlite() -> None:
    # The surrogate-backed in-memory search must still be byte-identical to the
    # SQLite backend (which stores the str file_id verbatim) for the same corpus.
    mem = _build_surrogate_index()
    sql = SQLiteHashIndex(":memory:")
    try:
        for file_id, code_offsets in _SURROGATE_CORPUS:
            sql.add(fp_with_hashes(file_id, code_offsets))
        query = fp_with_hashes("q" * 64, [(1000, 0), (1001, 0), (1002, 0), (1003, 0)])
        assert _search_tuples(mem, query, top_k=10) == _search_tuples(sql, query, top_k=10)
        assert mem.list_files() == sql.list_files()
        # The snapshot files map is parity-identical once normalized through
        # JSON (InMemory keeps tuples, SQL lists -- the round-trip the save/load
        # path always applies); compare via that canonical form.
        assert json.loads(json.dumps(mem.to_dict()["files"])) == \
               json.loads(json.dumps(sql.to_dict()["files"]))
    finally:
        sql.close()


def test_surrogate_per_posting_memory_drop() -> None:
    # MEASUREMENT: the surrogate (int, int) pair is far smaller than an
    # IndexPosting holding a 64-char file_id string. Compare the per-posting
    # footprint of the stored representation against the old representation.
    posting_tuple = (123, 45)  # (surrogate, time_offset) as stored now
    old_posting = IndexPosting(file_id="a" * 64, hash_code=1000, time_offset=45)
    file_id_str = "a" * 64

    # Old per-posting cost: the IndexPosting object PLUS the 64-char file_id it
    # references (the dominant cost the surrogate eliminates -- the str is held
    # once now, not per posting).
    old_bytes = sys.getsizeof(old_posting) + sys.getsizeof(file_id_str)
    new_bytes = sys.getsizeof(posting_tuple) + 2 * sys.getsizeof(123)
    assert new_bytes < old_bytes  # strictly smaller per posting

    # Whole-index sanity at a few-thousand-posting scale: building a 200-file x
    # 30-posting index stores 6000 compact pairs and zero per-posting file_id
    # strings (each 64-char id is held once in the surrogate maps).
    index = InMemoryHashIndex()
    for i in range(200):
        fid = f"{i:064x}"
        index.add(fp_with_hashes(fid, [(1000 + j, j) for j in range(30)]))
    assert index.posting_count == 6000
    # Exactly one file_id string per file is retained (in the surrogate map),
    # never one per posting.
    assert len(index._id_to_fid) == 200
    assert len(index._fid_to_id) == 200


# --------------------------------------------------------------------------- #
# Integer file_id surrogate key for the SQL backends (SQLiteHashIndex /
# PostgresHashIndex). Each posting row used to store the 64-char SHA-256 file_id
# verbatim; now a normalized files row maps file_id -> a small integer id and
# postings store that integer file_ref FK. These tests pin that the change is
# fully OUTPUT-PRESERVING (search rankings, to_dict snapshots, list_files,
# iter_metadata, counts byte-identical to the InMemory parity reference and to
# what the prior str-storing SQLite produced), that an OLD-schema .sqlite3
# migrates transparently, and they MEASURE the on-disk size win. Postgres is
# structural and gated behind @requires_pg (cannot be run live here).
# --------------------------------------------------------------------------- #

# Realistic 64-hex-char SHA-256-shaped file_ids (the value the surrogate replaces).
_SQL_SURROGATE_CORPUS = [
    ("a" * 64, [(1000, 10), (1001, 20), (1002, 30), (1003, 40)]),
    ("b" * 64, [(1000, 2), (1001, 40), (1004, 99), (1005, 125)]),
    ("c" * 64, [(1001, 21), (1002, 31), (1006, 7)]),
    ("d" * 64, [(1000, 11), (1001, 22), (1002, 33), (1003, 44), (1007, 88)]),
    ("e" * 64, []),  # zero-hash file: still tracked, still listed
]
_SQL_SURROGATE_QUERY = [(1000, 0), (1001, 0), (1002, 0), (1003, 0)]


def _build_old_schema_sqlite(path: Path, corpus) -> None:
    """Create a SQLite DB in the PRE-surrogate schema, exactly as the prior code.

    Mirrors the old ``SQLiteHashIndex`` layout (``postings.file_id TEXT``, a
    ``files`` table with no ``id`` column, the same two indexes) and the same
    signed-offset hash encoding and metadata JSON, built with raw sqlite3 so it
    is independent of the current (new-schema) class. This is the on-disk file an
    already-deployed index would have before the upgrade.
    """

    offset = SQLiteHashIndex._SIGNED_OFFSET
    conn = sqlite3.connect(str(path))
    conn.executescript(
        """
        CREATE TABLE files (file_id TEXT PRIMARY KEY, metadata TEXT NOT NULL);
        CREATE TABLE postings (
            file_id TEXT NOT NULL, hash_code INTEGER NOT NULL, time_offset INTEGER NOT NULL
        );
        CREATE INDEX idx_postings_hash ON postings(hash_code);
        CREATE INDEX idx_postings_file ON postings(file_id);
        """
    )
    for file_id, code_offsets in corpus:
        metadata = {
            "file_id": file_id, "path": f"/tmp/{file_id}", "handler": "test",
            "size_bytes": 10, "content_sha256": file_id,
            "hash_count": len(code_offsets), "landmark_count": 0,
        }
        conn.execute(
            "INSERT INTO files (file_id, metadata) VALUES (?, ?)",
            (file_id, json.dumps(metadata, sort_keys=True)),
        )
        conn.executemany(
            "INSERT INTO postings (file_id, hash_code, time_offset) VALUES (?, ?, ?)",
            [(file_id, code - offset, offset_value) for code, offset_value in code_offsets],
        )
    conn.commit()
    conn.close()


def _build_new_sqlite(corpus, database=":memory:") -> SQLiteHashIndex:
    index = SQLiteHashIndex(database)
    for file_id, code_offsets in corpus:
        index.add(fp_with_hashes(file_id, code_offsets))
    return index


def _build_in_memory(corpus) -> InMemoryHashIndex:
    index = InMemoryHashIndex()
    for file_id, code_offsets in corpus:
        index.add(fp_with_hashes(file_id, code_offsets))
    return index


def test_sql_surrogate_postings_store_integer_file_ref_not_string() -> None:
    # The footprint win: the new postings table stores an INTEGER file_ref FK,
    # never the 64-char file_id string; the files table holds the id<->file_id
    # surrogate mapping (one row per file, not per posting).
    index = _build_new_sqlite(_SQL_SURROGATE_CORPUS)
    posting_cols = {row[1] for row in index._conn.execute("PRAGMA table_info(postings)").fetchall()}
    file_cols = {row[1] for row in index._conn.execute("PRAGMA table_info(files)").fetchall()}
    assert "file_ref" in posting_cols and "file_id" not in posting_cols
    assert {"id", "file_id", "metadata"} <= file_cols
    # Every posting's file_ref is an int that resolves to a real files.id.
    valid_ids = {row[0] for row in index._conn.execute("SELECT id FROM files").fetchall()}
    refs = index._conn.execute("SELECT DISTINCT file_ref FROM postings").fetchall()
    assert refs and all(isinstance(r[0], int) and r[0] in valid_ids for r in refs)
    # files.id is the rowid surrogate; file_id stays the 64-char string.
    for row in index._conn.execute("SELECT id, file_id FROM files").fetchall():
        assert isinstance(row[0], int) and isinstance(row[1], str) and len(row[1]) == 64
    index.close()


def test_sql_surrogate_output_equivalent_to_in_memory_and_prior_sqlite() -> None:
    # HARD GATE 1 (OUTPUT-EQUIVALENCE): the new-schema SQLite backend's
    # search()/to_dict()/list_files()/iter_metadata()/posting_count/file_count
    # are byte-identical to the InMemory parity reference (the cross-backend
    # oracle the existing suite uses, which the prior str-storing SQLite was also
    # required to match -- so equality with it is equality with the prior SQLite
    # output). The dedicated migration test below independently proves an actual
    # OLD-schema database reads back identically. Includes the zero-hash file.
    new = _build_new_sqlite(_SQL_SURROGATE_CORPUS)
    mem = _build_in_memory(_SQL_SURROGATE_CORPUS)
    query = fp_with_hashes("q" * 64, _SQL_SURROGATE_QUERY)

    # search() ranking byte-identical to the in-memory parity reference.
    assert _search_tuples(new, query, top_k=10) == _search_tuples(mem, query, top_k=10)
    # Full SearchResult equality (score, votes, offset, confidence, metadata).
    assert new.search(query, top_k=10) == mem.search(query, top_k=10)
    # to_dict postings normalized through JSON (InMemory keeps tuples, SQL lists).
    assert json.loads(json.dumps(new.to_dict()["files"])) == \
           json.loads(json.dumps(mem.to_dict()["files"]))
    assert new.to_dict()["metadata"] == mem.to_dict()["metadata"]
    assert new.list_files() == mem.list_files()
    assert list(new.iter_metadata()) == list(mem.iter_metadata())
    assert new.posting_count == mem.posting_count
    assert new.file_count == mem.file_count

    # Replace-existing case: re-index a file_id, assert state matches a fresh
    # in-memory index built with the same replace, on both backends.
    replacement = [("a" * 64, [(2000, 7), (2001, 8)])]
    for fid, cos in replacement:
        new.add(fp_with_hashes(fid, cos))
        mem.add(fp_with_hashes(fid, cos))
    assert new.posting_count == mem.posting_count == (
        4 + 4 + 3 + 5 + 0 - 4 + 2  # corpus minus old "a" (4) plus new "a" (2)
    )
    assert _search_tuples(new, fp_with_hashes("q2" * 32, [(2000, 0), (2001, 0)]), top_k=10) == \
           _search_tuples(mem, fp_with_hashes("q2" * 32, [(2000, 0), (2001, 0)]), top_k=10)
    assert json.loads(json.dumps(new.to_dict()["files"]))["a" * 64] == [[2000, 7], [2001, 8]]
    new.close()


def test_sql_surrogate_cross_backend_parity_sqlite_inmemory_redis() -> None:
    # HARD GATE 2 (CROSS-BACKEND PARITY): InMemory == SQLite == Redis rankings
    # for the same surrogate corpus, plus snapshot interop in BOTH directions
    # (SQLite-written snapshot loads into every backend identically, and an
    # in-memory-written one loads back into the new-schema SQLite identically).
    query = fp_with_hashes("q" * 64, _SQL_SURROGATE_QUERY)
    reference: list[tuple[str, int, int, float]] | None = None
    for name, index in _parity_backends():
        for file_id, code_offsets in _SQL_SURROGATE_CORPUS:
            index.add(fp_with_hashes(file_id, code_offsets))
        tuples = _search_tuples(index, query, top_k=10)
        if reference is None:
            reference = tuples
        assert tuples == reference, name
        assert index.list_files() == [fid for fid, _ in sorted(_SQL_SURROGATE_CORPUS)], name


def test_sql_surrogate_snapshot_interop_both_directions(tmp_path: Path) -> None:
    # HARD GATE 2 cont.: snapshot interop both ways across the new SQLite schema.
    query = fp_with_hashes("q" * 64, _SQL_SURROGATE_QUERY)

    # SQLite -> snapshot -> InMemory (and back into a fresh new-schema SQLite).
    sql_source = _build_new_sqlite(_SQL_SURROGATE_CORPUS)
    snap = tmp_path / "from_sql.json"
    sql_source.save(snap)
    mem_target = InMemoryHashIndex.load(snap)
    sql_target = SQLiteHashIndex(":memory:").load_snapshot(snap)
    assert _search_tuples(mem_target, query) == _search_tuples(sql_source, query)
    assert _search_tuples(sql_target, query) == _search_tuples(sql_source, query)
    assert sql_target.to_dict()["files"] == sql_source.to_dict()["files"]
    sql_source.close()
    sql_target.close()

    # InMemory -> snapshot -> new-schema SQLite (reverse direction).
    mem_source = _build_in_memory(_SQL_SURROGATE_CORPUS)
    snap2 = tmp_path / "from_mem.json"
    mem_source.save(snap2)
    sql_from_mem = SQLiteHashIndex(":memory:").load_snapshot(snap2)
    assert _search_tuples(sql_from_mem, query) == _search_tuples(mem_source, query)
    assert sql_from_mem.list_files() == mem_source.list_files()
    assert json.loads(json.dumps(sql_from_mem.to_dict()["files"])) == \
           json.loads(json.dumps(mem_source.to_dict()["files"]))
    sql_from_mem.close()


def test_sqlite_migrates_old_schema_in_place_preserving_everything(tmp_path: Path) -> None:
    # HARD GATE 3 (MIGRATION): an OLD-schema .sqlite3 (postings.file_id TEXT,
    # files without id), built with raw sqlite3 to mimic the pre-change layout,
    # is opened by the new SQLiteHashIndex and must migrate in place to the
    # surrogate schema, returning search()/to_dict()/list_files() identical to a
    # freshly-built new-schema index of the same data.
    db = tmp_path / "legacy.sqlite3"
    _build_old_schema_sqlite(db, _SQL_SURROGATE_CORPUS)

    # Sanity: the file really is the OLD schema before we open it.
    probe = sqlite3.connect(str(db))
    old_cols = {row[1] for row in probe.execute("PRAGMA table_info(postings)").fetchall()}
    probe.close()
    assert "file_id" in old_cols and "file_ref" not in old_cols

    migrated = SQLiteHashIndex(db)
    # Schema was rewritten to the surrogate layout in place.
    new_cols = {row[1] for row in migrated._conn.execute("PRAGMA table_info(postings)").fetchall()}
    file_cols = {row[1] for row in migrated._conn.execute("PRAGMA table_info(files)").fetchall()}
    assert "file_ref" in new_cols and "file_id" not in new_cols
    assert "id" in file_cols

    fresh = _build_new_sqlite(_SQL_SURROGATE_CORPUS)
    query = fp_with_hashes("q" * 64, _SQL_SURROGATE_QUERY)
    assert _search_tuples(migrated, query, top_k=10) == _search_tuples(fresh, query, top_k=10)
    assert migrated.to_dict() == fresh.to_dict()  # postings order + metadata identical
    assert migrated.list_files() == fresh.list_files()
    assert migrated.posting_count == fresh.posting_count
    assert migrated.file_count == fresh.file_count
    # Every original metadata blob survived verbatim.
    for file_id, _ in _SQL_SURROGATE_CORPUS:
        assert migrated._metadata_for(file_id) == fresh._metadata_for(file_id)
    expected_files = fresh.list_files()
    expected_postings = fresh.posting_count
    fresh.close()
    migrated.close()

    # Re-opening the (now-migrated) file is a no-op: stable schema, same data.
    reopened = SQLiteHashIndex(db)
    new_cols2 = {row[1] for row in reopened._conn.execute("PRAGMA table_info(postings)").fetchall()}
    assert "file_ref" in new_cols2 and "file_id" not in new_cols2
    assert reopened.list_files() == expected_files
    assert reopened.posting_count == expected_postings
    reopened.close()


def test_sqlite_migration_is_writable_and_searchable_after_upgrade(tmp_path: Path) -> None:
    # After migrating, the index must behave like any new-schema index: add a new
    # file, remove an old one, and the surrogate path stays correct.
    db = tmp_path / "legacy_rw.sqlite3"
    _build_old_schema_sqlite(db, _SQL_SURROGATE_CORPUS)
    index = SQLiteHashIndex(db)

    index.add(fp_with_hashes("f" * 64, [(1000, 5), (1008, 6)]))
    index.remove("b" * 64)
    assert "f" * 64 in index.list_files()
    assert "b" * 64 not in index.list_files()
    # The added file is found, and no posting still references the removed file.
    assert index.search(fp_with_hashes("q" * 64, [(1008, 0)]), top_k=1)[0].file_id == "f" * 64
    assert all(p.file_id != "b" * 64 for p in index.query(1000))
    index.close()


def test_sqlite_fresh_db_is_not_treated_as_legacy(tmp_path: Path) -> None:
    # A brand-new database (no postings table yet) must NOT trip the migration
    # path; it just gets the new schema and works.
    db = tmp_path / "fresh.sqlite3"
    index = SQLiteHashIndex(db)
    index.add(fp_with_hashes("a" * 64, [(1, 1)]))
    cols = {row[1] for row in index._conn.execute("PRAGMA table_info(postings)").fetchall()}
    assert "file_ref" in cols
    assert index.list_files() == ["a" * 64]
    index.close()


def test_sqlite_surrogate_db_size_reduction_old_vs_new_schema(tmp_path: Path) -> None:
    # HARD GATE 4 (MEASURE): build ~2000 files x ~30 postings both ways and assert
    # the new-schema .sqlite3 is materially smaller than the old-schema one. The
    # win is that each of the 60000 posting rows now stores a small integer
    # file_ref instead of a 64-char file_id string.
    n_files, n_postings = 2000, 30
    offset = SQLiteHashIndex._SIGNED_OFFSET

    # OLD-schema DB built directly (the pre-change layout), WAL-checkpointed so
    # the measured size is the settled main database file, not WAL.
    old_db = tmp_path / "old.sqlite3"
    old_corpus = [(f"{i:064x}", [(1000 + j, j) for j in range(n_postings)]) for i in range(n_files)]
    conn = sqlite3.connect(str(old_db))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(
        """
        CREATE TABLE files (file_id TEXT PRIMARY KEY, metadata TEXT NOT NULL);
        CREATE TABLE postings (
            file_id TEXT NOT NULL, hash_code INTEGER NOT NULL, time_offset INTEGER NOT NULL
        );
        CREATE INDEX idx_postings_hash ON postings(hash_code);
        CREATE INDEX idx_postings_file ON postings(file_id);
        """
    )
    for file_id, code_offsets in old_corpus:
        meta = {
            "file_id": file_id, "path": "", "handler": "test", "size_bytes": 0,
            "content_sha256": file_id, "hash_count": n_postings, "landmark_count": 0,
        }
        conn.execute("INSERT INTO files VALUES (?, ?)", (file_id, json.dumps(meta, sort_keys=True)))
        conn.executemany(
            "INSERT INTO postings VALUES (?, ?, ?)",
            [(file_id, code - offset, off) for code, off in code_offsets],
        )
    conn.commit()
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    conn.commit()
    conn.close()
    old_size = old_db.stat().st_size

    # NEW-schema DB via the class (add_many for the one-transaction bulk path).
    new_db = tmp_path / "new.sqlite3"
    index = SQLiteHashIndex(new_db)
    index.add_many([fp_with_hashes(fid, cos) for fid, cos in old_corpus])
    index._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    index._conn.commit()
    new_size = new_db.stat().st_size

    assert index.posting_count == n_files * n_postings
    assert index.file_count == n_files
    index.close()

    # The surrogate schema is materially smaller -- conservatively assert a >=25%
    # reduction (the observed win is far larger; the measurement is reported in
    # the task summary). This guards the optimization from regressing.
    assert new_size < old_size
    reduction = (old_size - new_size) / old_size
    assert reduction >= 0.25, f"old={old_size} new={new_size} reduction={reduction:.3f}"


@requires_pg
def test_pg_surrogate_output_equivalent_to_in_memory(pg_index) -> None:
    # HARD GATE 1/2 for Postgres (structural; only runs with a live server):
    # the surrogate-schema Postgres backend's rankings, snapshot, enumeration and
    # counts match the in-memory parity reference for the same corpus.
    mem = _build_in_memory(_SQL_SURROGATE_CORPUS)
    for file_id, code_offsets in _SQL_SURROGATE_CORPUS:
        pg_index.add(fp_with_hashes(file_id, code_offsets))
    query = fp_with_hashes("q" * 64, _SQL_SURROGATE_QUERY)

    assert _search_tuples(pg_index, query, top_k=10) == _search_tuples(mem, query, top_k=10)
    assert pg_index.list_files() == mem.list_files()
    assert list(pg_index.iter_metadata()) == list(mem.iter_metadata())
    assert pg_index.posting_count == mem.posting_count
    assert pg_index.file_count == mem.file_count
    # The postings table carries an integer file_ref, not the file_id string.
    with pg_index._conn.cursor() as cur:
        cur.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name = %s",
            (pg_index._postings_table,),
        )
        cols = {row[0] for row in cur.fetchall()}
    pg_index._conn.rollback()
    assert "file_ref" in cols and "file_id" not in cols


@requires_pg
def test_pg_migrates_old_schema_in_place(pg_index) -> None:
    # HARD GATE 3 for Postgres (structural; only runs with a live server): drop
    # the surrogate tables the fixture created, recreate the OLD schema with
    # data, then re-open a PostgresHashIndex on the same prefix and assert it
    # migrated to the surrogate layout with output identical to the in-memory
    # reference for the same data.
    files_t, posts_t = pg_index._files_table, pg_index._postings_table
    offset = PostgresHashIndex._SIGNED_OFFSET
    with pg_index._conn.cursor() as cur:
        cur.execute(f"DROP TABLE IF EXISTS {posts_t}, {files_t}")
        cur.execute(f"CREATE TABLE {files_t} (file_id TEXT PRIMARY KEY, metadata JSONB NOT NULL)")
        cur.execute(
            f"CREATE TABLE {posts_t} "
            "(file_id TEXT NOT NULL, hash_code BIGINT NOT NULL, time_offset INTEGER NOT NULL)"
        )
        for file_id, code_offsets in _SQL_SURROGATE_CORPUS:
            meta = {
                "file_id": file_id, "path": f"/tmp/{file_id}", "handler": "test",
                "size_bytes": 10, "content_sha256": file_id,
                "hash_count": len(code_offsets), "landmark_count": 0,
            }
            cur.execute(
                f"INSERT INTO {files_t} (file_id, metadata) VALUES (%s, %s::jsonb)",
                (file_id, json.dumps(meta, sort_keys=True)),
            )
            for code, off in code_offsets:
                cur.execute(
                    f"INSERT INTO {posts_t} (file_id, hash_code, time_offset) VALUES (%s, %s, %s)",
                    (file_id, code - offset, off),
                )
    pg_index._conn.commit()

    migrated = PostgresHashIndex(dsn=PG_DSN, table_prefix="fp_pytest")  # same prefix -> migrates
    try:
        with migrated._conn.cursor() as cur:
            cur.execute(
                "SELECT column_name FROM information_schema.columns WHERE table_name = %s",
                (posts_t,),
            )
            cols = {row[0] for row in cur.fetchall()}
        migrated._conn.rollback()
        assert "file_ref" in cols and "file_id" not in cols

        mem = _build_in_memory(_SQL_SURROGATE_CORPUS)
        query = fp_with_hashes("q" * 64, _SQL_SURROGATE_QUERY)
        assert _search_tuples(migrated, query, top_k=10) == _search_tuples(mem, query, top_k=10)
        assert migrated.list_files() == mem.list_files()
        assert migrated.posting_count == mem.posting_count
    finally:
        migrated.close()
