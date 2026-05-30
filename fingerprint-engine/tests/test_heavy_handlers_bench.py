"""Fast, gated regression tests that pin the heavy-handler benchmark headline.

These lock in the *working* behaviour the benchmark in
``benchmarks/heavy_handlers.py`` measures, so the video and embedding skeletons
keep fingerprinting end-to-end:

* video (gated on ``av`` + ``imageio``): self recall@1 == 1.0, and at least one
  cadence-preserving near-duplicate variant ranks its parent #1;
* embedding (numpy-only, always runs): self recall@1 == 1.0, and an append /
  edit near-duplicate ranks its parent #1, while a length-changing variant that
  crosses the adaptive-window boundary is *expected* to miss (the documented
  design gap is pinned, not hidden).

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

from benchmarks.heavy_handlers import run_embedding_benchmark, run_video_benchmark

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


def test_embedding_benchmark_pins_adaptive_window_gap() -> None:
    # The handler sets no class default_signal_window, so the signal is
    # fingerprinted at the LENGTH-ADAPTIVE global window, not a d-aligned one.
    # A large append crosses an adaptive-window boundary (512 -> 1024) and lands
    # on a different time grid, so it FAILS to align. Pin that documented gap so
    # a future change that fixes (or regresses) it is visible.
    report = run_embedding_benchmark(num_docs=8, num_vectors=60, dim=64)
    by_variant = {row["variant"]: row for row in report["per_variant"]}
    assert by_variant["append_30"]["matched"] is False
    assert by_variant["append_30"]["effective_window"] != by_variant["append_5"]["effective_window"]


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
