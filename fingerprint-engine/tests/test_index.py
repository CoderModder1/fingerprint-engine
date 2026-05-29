from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.index import InMemoryHashIndex, RedisHashIndex, SQLiteHashIndex
from core.models import ConstellationHash, Fingerprint


def _fake_redis():
    fakeredis = pytest.importorskip("fakeredis")
    return fakeredis.FakeStrictRedis(decode_responses=True)


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


def test_sqlite_file_backend_persists(tmp_path: Path) -> None:
    # A file-backed SQLite index must survive being reopened.
    db = tmp_path / "index.sqlite3"
    fingerprint = make_fingerprint("persisted", [5, 6, 7])
    SQLiteHashIndex(db).add(fingerprint)

    reopened = SQLiteHashIndex(db)
    assert reopened.file_count == 1
    assert reopened.posting_count == 3
    assert reopened.search(fingerprint)[0].file_id == "persisted"
