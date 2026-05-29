from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.index import InMemoryHashIndex
from core.models import ConstellationHash, Fingerprint


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
