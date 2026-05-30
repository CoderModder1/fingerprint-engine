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
from fingerprint_engine.core.models import FingerprintConfig  # noqa: E402 - after sys.path bootstrap

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


# ---------------------------------------------------------------------------
# OPT-IN multi-resolution window bank: the key experiment. The default single-
# window matcher scores ~0 on audio excerpts/clips (whole-signal re-normalisation
# shifts the global time grid). Fingerprinting at a BANK of windows -- with the
# window folded into each hash so window-w only collides with window-w -- lets a
# query align at whatever window survives the excerpt, fixing that 0-recall while
# leaving exact recall and the impostor separation intact (at an N-fold posting
# cost). These tests pin the OFF->ON improvement and the no-regression guards.
# ---------------------------------------------------------------------------

_AUDIO_WINDOW_BANK = (512, 1024, 2048, 4096)


def test_window_bank_fixes_audio_excerpt_recall(tmp_path: Path) -> None:
    pytest.importorskip("scipy")
    bank_cfg = FingerprintConfig(window_bank=_AUDIO_WINDOW_BANK)

    # OFF: the documented baseline -- excerpt/clip recall is at the floor.
    off = evaluate_audio(Fingerprinter(), np.random.default_rng(SEED), AUDIO_CORPUS, tmp_path)
    off_clip = _mutation(off, "clip_prefix_60pct")["recall_at_1"]
    off_excerpt = _mutation(off, "excerpt_mid")["recall_at_1"]

    # ON: the same corpus and mutations, fingerprinted at the window bank.
    on = evaluate_audio(
        Fingerprinter(config=bank_cfg), np.random.default_rng(SEED), AUDIO_CORPUS, tmp_path
    )
    on_clip = _mutation(on, "clip_prefix_60pct")["recall_at_1"]
    on_excerpt = _mutation(on, "excerpt_mid")["recall_at_1"]

    # The bank must strictly improve at least one excerpt/clip case, and not
    # regress the other -- the headline result.
    assert (on_clip > off_clip) or (on_excerpt > off_excerpt)
    assert on_clip >= off_clip
    assert on_excerpt >= off_excerpt

    # No-regression guards: exact self-match and the true-vs-impostor separation
    # must survive the bank (it adds resolutions, it does not blur identity).
    assert on.exact_recall_at_1 == 1.0
    assert on.mean_true_confidence >= 0.5
    assert on.mean_impostor_confidence < 0.05

    # The bank ~N-folds postings (one constellation per window). Confirm the cost
    # is real and bounded near the bank size, not a free lunch.
    multiplier = on.avg_hashes_per_file / max(1.0, off.avg_hashes_per_file)
    assert multiplier > 1.5


def test_window_bank_preserves_text_recall(tmp_path: Path) -> None:
    # The bank must not regress the already-strong text near-dup recall while it
    # opens the cross-length resolutions.
    bank_cfg = FingerprintConfig(window_bank=_AUDIO_WINDOW_BANK)
    rng = np.random.default_rng(SEED)
    report, _index, _fps = evaluate_text(Fingerprinter(config=bank_cfg), rng, TEXT_CORPUS, tmp_path)
    assert report.exact_recall_at_1 == 1.0
    for name in ("append", "truncate_prefix", "front_insert", "char_edit_2pct", "char_edit_5pct"):
        assert _mutation(report, name)["recall_at_1"] == 1.0, f"{name} regressed under the bank"
    assert report.mean_impostor_confidence < 0.05
