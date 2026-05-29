from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.index import (
    InMemoryHashIndex,
    PostgresHashIndex,
    RedisHashIndex,
    SQLiteHashIndex,
)
from core.models import Calibration, ConstellationHash, Fingerprint


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


def test_time_coherent_search_ranks_aligned_match_first() -> None:
    index = InMemoryHashIndex()
    index.add(make_fingerprint("aligned", [10, 20, 30, 40]))
    index.add(make_fingerprint("scattered", [2, 40, 99, 125]))
    query = make_fingerprint("query", [3, 13, 23, 33])

    results = index.search(query, top_k=2)

    assert results[0].file_id == "aligned"
    assert results[0].aligned_votes == 4
    assert results[0].offset == 7


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


def test_sqlite_file_backend_persists(tmp_path: Path) -> None:
    # A file-backed SQLite index must survive being reopened.
    db = tmp_path / "index.sqlite3"
    fingerprint = make_fingerprint("persisted", [5, 6, 7])
    SQLiteHashIndex(db).add(fingerprint)

    reopened = SQLiteHashIndex(db)
    assert reopened.file_count == 1
    assert reopened.posting_count == 3
    assert reopened.search(fingerprint)[0].file_id == "persisted"
