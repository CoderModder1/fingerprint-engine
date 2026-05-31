"""Opt-in matching: offset-tolerant voting and the candidate_limit prefilter."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
from _fixtures import (
    fp_with_hashes,
    make_fingerprint,
)
from _fixtures import (
    parity_backends as _parity_backends,
)
from _fixtures import (
    search_tuples as _search_tuples,
)

from fingerprint_engine.core.index import (
    SNAPSHOT_SCHEMA_VERSION,
    InMemoryHashIndex,
)
from fingerprint_engine.core.models import (
    Calibration,
    Fingerprint,
)

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


def test_candidate_limit_keeps_repeated_code_high_vote_match_across_backends() -> None:
    # A6 regression: a match dominated by ONE code repeated at many COHERENT
    # offsets has huge ALIGNED votes but few DISTINCT shared codes. The old
    # prefilter counted distinct codes (+1 per code), UNDER-counting such a match,
    # so a tight candidate_limit dropped the true #1 below low-vote decoys that
    # merely share more distinct codes. Ranking on shared POSTINGS (query x file
    # multiplicity = a true upper bound on aligned votes) keeps it.
    coherent = [(1, off) for off in range(0, 1000, 50)]  # code 1 at 20 offsets
    query = fp_with_hashes(
        "q", coherent + [(2, 5), (3, 6), (10, 7), (11, 8), (12, 9), (13, 10), (14, 11)]
    )
    # TRUE: code 1 at those 20 offsets shifted by a constant delta (-> ~20 aligned
    # votes), plus codes 2,3 -> only 3 DISTINCT shared codes.
    true_file = fp_with_hashes(
        "true", [(1, off + 100) for off in range(0, 1000, 50)] + [(2, 500), (3, 600)]
    )
    # 8 decoys each share 5 DISTINCT query codes (10..14), so under the old
    # distinct-code ranking they (5) out-ranked TRUE (3) and a limit of 5 dropped
    # it. They share only 5 postings, far below TRUE's ~402.
    decoys = [
        fp_with_hashes(f"decoy{i}", [(10, 0), (11, 0), (12, 0), (13, 0), (14, 0)])
        for i in range(8)
    ]
    for name, index in _parity_backends():
        for fp in [true_file, *decoys]:
            index.add(fp)
        # candidate_limit (5) < decoy count (8): under distinct-code ranking the
        # decoys would fill the candidate set and exclude TRUE. It must survive.
        top = index.search(query, top_k=1, candidate_limit=5)
        assert top and top[0].file_id == "true", name
        assert top[0].aligned_votes >= 20, (name, top[0].aligned_votes)


def test_from_dict_counts_dropped_postings_and_recomputes_hash_count(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # A7: from_dict copies snapshot metadata wholesale, so a dropped posting left
    # a STALE hash_count that inflates the confidence denominator (deflating every
    # match for that file). It must drop + WARN + recompute hash_count to the
    # postings actually loaded, while leaving a clean file untouched.
    data = {
        "backend": "in_memory",
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "files": {
            # 3 valid postings + a wrong-arity entry + a non-list entry (2 dropped).
            "file-a": [[1000, 0], [1001, 0], [1002, 0], [1003], "garbage"],
            "file-b": [[2000, 0], [2001, 0]],  # clean
        },
        "metadata": {
            "file-a": {"file_id": "file-a", "handler": "test", "hash_count": 5},  # STALE
            "file-b": {"file_id": "file-b", "handler": "test", "hash_count": 2},
        },
    }
    with caplog.at_level(logging.WARNING):
        index = InMemoryHashIndex.from_dict(data)

    # Degraded file's hash_count recomputed to the kept count; clean file untouched.
    assert index._metadata["file-a"]["hash_count"] == 3
    assert index._metadata["file-b"]["hash_count"] == 2
    messages = [r.getMessage() for r in caplog.records]
    assert any("skipped 2" in m for m in messages)
    assert any("degraded" in m for m in messages)

    # Confidence is calibrated to the recomputed count: a query of all 5 intended
    # codes aligns the 3 survivors -> 3 / min(5, 3) = 1.0, not the deflated
    # 3 / min(5, 5) = 0.6 the stale hash_count would have produced.
    query = fp_with_hashes("q", [(1000, 0), (1001, 0), (1002, 0), (1003, 0), (1004, 0)])
    result = index.search(query, top_k=1)[0]
    assert result.file_id == "file-a"
    assert result.aligned_votes == 3
    assert result.confidence == pytest.approx(1.0)


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


