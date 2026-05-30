"""Fast, gated regression tests that pin the heavy-handler benchmark headline.

These lock in the *working* behaviour the benchmark in
``benchmarks/heavy_handlers.py`` measures, so the video and embedding skeletons
keep fingerprinting end-to-end:

* video (gated on ``av`` + ``imageio``): self recall@1 == 1.0, and at least one
  cadence-preserving near-duplicate variant ranks its parent #1;
* embedding (numpy-only, always runs): self recall@1 == 1.0, append / edit
  near-duplicates rank their parent #1, and a LENGTH-CROSSING near-duplicate
  (a big append) also ranks its parent #1 -- the per-frame window=d makes
  matching length-stable -- while a DIFFERENT-dimensionality copy of the same
  content correctly does NOT match (window=d isolates by dimensionality).

The benchmark module is imported and reused so the tests exercise exactly the
synthesis + indexing the reported numbers come from. Everything is seeded and
tiny, so each test runs in well under a few seconds.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.heavy_handlers import (
    run_embedding_benchmark,
    run_encoder_benchmark,
    run_video_benchmark,
)

# ---------------------------------------------------------------------------
# Embedding: numpy-only, always runs.
# ---------------------------------------------------------------------------


def test_embedding_benchmark_self_recall_is_perfect() -> None:
    report = run_embedding_benchmark(num_docs=8, num_vectors=60, dim=64)
    assert report["self_recall_at_1"] == 1.0
    assert report["mean_self_confidence"] == 1.0


def test_embedding_benchmark_append_and_edit_variants_match() -> None:
    report = run_embedding_benchmark(num_docs=8, num_vectors=60, dim=64)
    by_variant = {row["variant"]: row for row in report["per_variant"]}

    # An append that stays within the same adaptive window, and a small
    # row-perturbation, must both still rank the parent #1.
    assert by_variant["append_5"]["matched"] is True
    assert by_variant["perturb_5_rows"]["matched"] is True
    # A few row edits keep the parent at rank 1 even though confidence drops.
    assert by_variant["insert_3_rows"]["matched"] is True

    # Impostor separation: an unrelated doc must not out-rank a real near-dup's
    # confidence, so the spurious ceiling sits below the matching append.
    assert report["max_impostor_confidence"] < by_variant["append_5"]["confidence"]


def test_embedding_benchmark_length_stable_and_dim_isolated() -> None:
    # The handler overrides extract_peaks to fingerprint at a per-frame
    # window=hop=d (fixed_window=True), so all same-d files share one time grid
    # regardless of sequence length -> length-STABLE matching. The big-append
    # near-dup (append_30) crosses what was an adaptive-window boundary at the
    # GLOBAL pipeline (effective_window 512 -> 1024) yet must now still rank the
    # true parent #1, because the actual fingerprint window is d, not the global
    # adaptive window.
    report = run_embedding_benchmark(num_docs=8, num_vectors=60, dim=64)
    by_variant = {row["variant"]: row for row in report["per_variant"]}

    # Length stability: the length-crossing near-dup now MATCHES its parent.
    assert by_variant["append_30"]["matched"] is True
    # It really did cross the global adaptive-window boundary (proving the old
    # gap), yet it still matched -- the d window, not the global window, governs.
    assert by_variant["append_30"]["effective_window"] != by_variant["append_5"]["effective_window"]
    # Every same-d file -- original or near-dup -- was fingerprinted at window=d.
    assert by_variant["append_30"]["embedding_window"] == 64
    assert by_variant["append_5"]["embedding_window"] == 64

    # Self recall stays perfect.
    assert report["self_recall_at_1"] == 1.0

    # Dimensionality isolation: the SAME base content re-embedded at 2*d lands on
    # a different time grid (window=2*d) and must NOT match the d=64 parent.
    assert by_variant["different_dim"]["embedding_window"] == 128
    assert by_variant["different_dim"]["matched"] is False


# ---------------------------------------------------------------------------
# Encoder: gated on model2vec (the real encode-on-load path). Skips cleanly when
# model2vec is absent, so CI never downloads a model.
# ---------------------------------------------------------------------------


def test_encoder_benchmark_exact_reuse_matches_and_self_recall_perfect() -> None:
    pytest.importorskip("model2vec")

    report = run_encoder_benchmark(num_docs=8)
    if report.get("skipped"):
        pytest.skip(f"encoder backend unavailable: {report.get('reason')}")

    # The real model2vec encode-on-load path produces matchable fingerprints:
    # every indexed doc ranks itself #1 with full confidence.
    assert report["self_recall_at_1"] == 1.0
    assert report["mean_self_confidence"] == 1.0
    # potion-base-8M is a 256-dim static model; each vector is exactly one frame,
    # so the per-frame fingerprint window equals d.
    assert report["embedding_dim"] == 256

    by_variant = {row["variant"]: row for row in report["per_variant"]}

    # EXACT_PASSAGE_REUSE: a doc that copies several of doc-0's paragraphs
    # VERBATIM ranks doc-0 #1. Verbatim chunks encode to bit-identical vectors ->
    # identical spectra -> identical constellation hashes, so it shares many hash
    # codes and clears the impostor ceiling by a wide margin -- this is the core
    # claim, that the encoder path matches SHARED EXACT PASSAGES.
    reuse = by_variant["exact_passage_reuse"]
    assert reuse["matched"] is True
    assert reuse["shared_hash_codes"] > 0
    assert reuse["confidence"] > report["max_impostor_confidence"]


def test_encoder_benchmark_paraphrase_is_not_a_semantic_match() -> None:
    # The documented LIMIT, pinned because model2vec static embeddings are
    # deterministic. A FULL paraphrase (every paragraph reworded, no verbatim
    # line shared) is semantically close but encodes to DIFFERENT vectors, so it
    # shares essentially no hash codes and its confidence collapses to the
    # impostor/noise floor. The encoder path detects SHARED EXACT EMBEDDING
    # SUB-SEQUENCES, not semantic paraphrase.
    pytest.importorskip("model2vec")

    report = run_encoder_benchmark(num_docs=8)
    if report.get("skipped"):
        pytest.skip(f"encoder backend unavailable: {report.get('reason')}")

    by_variant = {row["variant"]: row for row in report["per_variant"]}
    paraphrase = by_variant["paraphrase_all"]
    reuse = by_variant["exact_passage_reuse"]

    # The paraphrase shares far fewer exact hash codes than verbatim reuse, and
    # its confidence sits at or below the impostor ceiling -- indistinguishable
    # from an unrelated doc, i.e. NOT a real match even though it nominally
    # rank-1s doc-0 in this tiny same-pool corpus.
    assert paraphrase["shared_hash_codes"] < reuse["shared_hash_codes"]
    assert paraphrase["confidence"] <= report["max_impostor_confidence"]
    assert paraphrase["confidence"] < reuse["confidence"]


# ---------------------------------------------------------------------------
# Video: gated on the decode backend + writer being present.
# ---------------------------------------------------------------------------


def test_video_benchmark_self_recall_and_a_near_dup_match() -> None:
    pytest.importorskip("av")
    pytest.importorskip("imageio")
    pytest.importorskip("PIL")

    report = run_video_benchmark(num_clips=6, num_frames=48, width=64, height=48)
    if report.get("skipped"):
        pytest.skip(f"video backend unavailable: {report.get('reason')}")

    assert report["self_recall_at_1"] == 1.0
    assert report["mean_self_confidence"] == 1.0

    by_variant = {row["variant"]: row for row in report["per_variant"]}
    # The cadence-preserving re-encode and the middle excerpt rank the parent
    # clip #1 (their keyframe grid stays aligned with the original's).
    assert by_variant["reencode_bitrate"]["matched"] is True
    assert by_variant["excerpt_10_40"]["matched"] is True
    # Matching near-dups clear the impostor ceiling, so rank-1 recall is a real
    # signal and not a coin flip -- even though the absolute confidence is low.
    assert by_variant["reencode_bitrate"]["confidence"] > report["max_impostor_confidence"]
