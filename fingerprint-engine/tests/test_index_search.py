"""Search ranking, normalized confidence, and calibration filtering."""

from __future__ import annotations

from _fixtures import (
    make_fingerprint,
)

from fingerprint_engine.core.index import (
    InMemoryHashIndex,
    SQLiteHashIndex,
)
from fingerprint_engine.core.models import (
    Calibration,
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


