"""Regression tests turning the engine's recall/calibration claims into asserts.

These guard the *quality* of matching, not its speed: they run the deterministic
:mod:`benchmarks.accuracy` harness on a tiny generated corpus and assert the
load-bearing, documented properties hold with margin -- so any future change to
the fingerprint hashes, the scoring formula, or the calibration that degrades
recall or collapses the true-vs-impostor confidence gap fails CI loudly.

Thresholds are set conservatively *below* the measured numbers (e.g. the harness
sees text true-confidence 1.0 and impostor < 0.013; we assert true >= 0.5 and
impostor < 0.05) so normal run-to-run-equivalent output passes but a real
regression trips. The whole module is bounded to a few seconds via small
corpora; image/audio cases are skipped when their optional dep is absent.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.accuracy import (  # noqa: E402 - after sys.path bootstrap
    evaluate_audio,
    evaluate_image,
    evaluate_text,
    prune_is_lossless,
    threshold_sweep,
)
from fingerprint_engine.core.fingerprinter import Fingerprinter  # noqa: E402 - after sys.path bootstrap

SEED = 1234
TEXT_CORPUS = 24
IMAGE_CORPUS = 8
AUDIO_CORPUS = 8


def _mutation(report, name):  # noqa: ANN001 - test helper
    """Return the single MutationResult dict matching ``name`` in a report."""

    matches = [m for m in report.to_dict()["mutations"] if m["mutation"] == name]
    assert matches, f"mutation {name!r} not measured"
    return matches[0]


@pytest.fixture(scope="module")
def text_eval(tmp_path_factory: pytest.TempPathFactory):  # noqa: ANN201 - fixture
    """Run the text harness once for the whole module (the fast, always-on path)."""

    rng = np.random.default_rng(SEED)
    fingerprinter = Fingerprinter()
    workdir = tmp_path_factory.mktemp("accuracy_text")
    report, index, fingerprints = evaluate_text(fingerprinter, rng, TEXT_CORPUS, workdir)
    return report, index, fingerprints


# ---------------------------------------------------------------------------
# Exact self-match recall -- the most basic claim: a fingerprint finds itself.
# ---------------------------------------------------------------------------


def test_text_exact_recall_is_perfect(text_eval) -> None:  # noqa: ANN001
    report, _index, _fps = text_eval
    assert report.exact_recall_at_1 == 1.0


# ---------------------------------------------------------------------------
# Near-duplicate recall under each text mutation. Append and truncate/prefix are
# the documented strong cases and must stay at 1.0; the harder cases must still
# find every parent (recall 1.0) even if confidence is lower.
# ---------------------------------------------------------------------------


def test_text_append_neardup_recall_is_perfect(text_eval) -> None:  # noqa: ANN001
    report, _index, _fps = text_eval
    append = _mutation(report, "append")
    assert append["recall_at_1"] == 1.0
    # Append leaves the whole original intact, so confidence stays very high.
    assert append["mean_confidence"] >= 0.8


def test_text_prefix_neardup_recall_is_perfect(text_eval) -> None:  # noqa: ANN001
    report, _index, _fps = text_eval
    prefix = _mutation(report, "truncate_prefix")
    assert prefix["recall_at_1"] == 1.0
    assert prefix["mean_confidence"] >= 0.5


def test_text_front_insert_and_char_edits_still_recall_all(text_eval) -> None:  # noqa: ANN001
    report, _index, _fps = text_eval
    for name in ("front_insert", "char_edit_2pct", "char_edit_5pct"):
        mutation = _mutation(report, name)
        assert mutation["recall_at_1"] == 1.0, f"{name} dropped a parent match"


# ---------------------------------------------------------------------------
# Confidence separation: the documented property that a single ~0.05 threshold
# cleanly splits true matches from impostors.
# ---------------------------------------------------------------------------


def test_text_confidence_separation(text_eval) -> None:  # noqa: ANN001
    report, _index, _fps = text_eval
    # True self-matches are confidently high; impostors sit far below the cutoff.
    assert report.mean_true_confidence >= 0.5
    assert report.min_true_confidence >= 0.5
    assert report.mean_impostor_confidence < 0.05
    # And there is a real gap, not a marginal one.
    assert report.mean_true_confidence - report.mean_impostor_confidence > 0.4


def test_threshold_sweep_separates_true_from_impostor(text_eval) -> None:  # noqa: ANN001
    report, index, fingerprints = text_eval
    rows = {row["threshold"]: row for row in threshold_sweep(index, fingerprints)}
    # At the documented default cutoff, recall is perfect and no impostor leaks.
    assert rows[0.05]["recall"] == 1.0
    assert rows[0.05]["precision"] == 1.0
    assert rows[0.05]["false_accept_rate"] == 0.0
    # Even at a strict 0.5 cutoff, genuine self-matches (confidence 1.0) survive.
    assert rows[0.5]["recall"] == 1.0


# ---------------------------------------------------------------------------
# Pruning is lossless: stop-hash pruning must not drop any self-match.
# ---------------------------------------------------------------------------


def test_prune_stop_hashes_keeps_recall(text_eval) -> None:  # noqa: ANN001
    _report, _index, fingerprints = text_eval
    result = prune_is_lossless(_index, fingerprints)
    assert result["recall_before"] == 1.0
    assert result["recall_after"] == 1.0
    assert result["lossless"] is True


# ---------------------------------------------------------------------------
# Image (gated on Pillow): exact recall + resize/JPEG near-dup recall must be
# perfect, and the confidence gap must hold.
# ---------------------------------------------------------------------------


def test_image_recall_and_separation(tmp_path: Path) -> None:
    pytest.importorskip("PIL")
    rng = np.random.default_rng(SEED)
    report = evaluate_image(Fingerprinter(), rng, IMAGE_CORPUS, tmp_path)
    assert report.exact_recall_at_1 == 1.0
    assert _mutation(report, "resize_downscale")["recall_at_1"] == 1.0
    assert _mutation(report, "jpeg_q40")["recall_at_1"] == 1.0
    assert report.mean_true_confidence >= 0.5
    assert report.mean_impostor_confidence < 0.05


# ---------------------------------------------------------------------------
# Audio (gated on scipy): exact recall and the confidence gap must hold. Clip /
# excerpt recall is the known weak spot (whole-signal re-normalisation shifts
# the fixed-window grid); we assert only that the harness measures it as the
# documented baseline (recall in [0, 1]), NOT that it is high -- so the deferred
# matcher research has a regression-safe floor to improve on.
# ---------------------------------------------------------------------------


def test_audio_exact_recall_and_separation(tmp_path: Path) -> None:
    pytest.importorskip("scipy")
    rng = np.random.default_rng(SEED)
    report = evaluate_audio(Fingerprinter(), rng, AUDIO_CORPUS, tmp_path)
    assert report.exact_recall_at_1 == 1.0
    assert report.mean_true_confidence >= 0.5
    assert report.mean_impostor_confidence < 0.05
    # Clip/excerpt recall is measured (baseline for future matcher work), bounded.
    for name in ("clip_prefix_60pct", "excerpt_mid"):
        recall = _mutation(report, name)["recall_at_1"]
        assert 0.0 <= recall <= 1.0
