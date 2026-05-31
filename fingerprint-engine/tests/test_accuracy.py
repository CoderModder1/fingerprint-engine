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
    _false_accept_rate,
    _hard_text_corpus,
    _hard_text_mutations,
    _mutation_recall,
    confusability_precision,
    evaluate_audio,
    evaluate_image,
    evaluate_text,
    prune_is_lossless,
    sweep_audio_flags,
    sweep_image_flags,
    threshold_sweep,
)
from fingerprint_engine.core.fingerprinter import Fingerprinter  # noqa: E402 - after sys.path bootstrap
from fingerprint_engine.core.models import FingerprintConfig  # noqa: E402 - after sys.path bootstrap

SEED = 1234
TEXT_CORPUS = 24
IMAGE_CORPUS = 8
AUDIO_CORPUS = 8

# Small hard-corpus sizes -- kept tiny so each hard test stays a fraction of a
# second (the discriminating power, not statistical precision, is what's pinned).
HARD_TEXT_CORPUS = 16
HARD_IMAGE_CORPUS = 8
HARD_AUDIO_CORPUS = 6


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
    pytest.importorskip("PIL", exc_type=ImportError)
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
    pytest.importorskip("scipy", exc_type=ImportError)
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


def test_audio_excerpt_recall_works_by_default(tmp_path: Path) -> None:
    pytest.importorskip("scipy", exc_type=ImportError)

    # v2: the audio handler fingerprints with a multi-resolution window bank BY
    # DEFAULT, so clip/excerpt matching works out of the box -- no opt-in needed.
    default = evaluate_audio(Fingerprinter(), np.random.default_rng(SEED), AUDIO_CORPUS, tmp_path)
    assert _mutation(default, "clip_prefix_60pct")["recall_at_1"] == 1.0
    assert _mutation(default, "excerpt_mid")["recall_at_1"] == 1.0

    # No-regression: exact self-match and the true-vs-impostor separation hold
    # (the bank adds resolutions, it does not blur identity).
    assert default.exact_recall_at_1 == 1.0
    assert default.mean_true_confidence >= 0.5
    assert default.mean_impostor_confidence < 0.05

    # The bank is genuinely active by default: it ~N-folds postings vs a
    # single-resolution audio config, so the default carries the documented cost.
    single = evaluate_audio(
        Fingerprinter(config=FingerprintConfig(window_bank=(4096,))),
        np.random.default_rng(SEED),
        AUDIO_CORPUS,
        tmp_path,
    )
    assert default.avg_hashes_per_file > 1.5 * single.avg_hashes_per_file


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


# ---------------------------------------------------------------------------
# HARD corpus: the discriminating-power gate. The standard matrix above saturates
# text/image recall at 1.0 even on heavy mutations, so an opt-in flag's effect on
# RECALL is invisible there. The hard corpus (benchmarks.accuracy hard mode) is
# built to push default recall BELOW 1.0 and to inject false-accept pressure, so
# a flag's recall win and precision cost are both measurable. These tests pin that
# new power: (a) the default config really does drop below 1.0 on the hardest
# mutations, and (b) the specific flags that recover recall actually do so. If a
# future change re-saturates recall (re-hiding the flags) OR breaks a flag's
# recovery, these trip. They use tiny corpora to stay sub-second.
# ---------------------------------------------------------------------------


def _hard_text(tmp_path: Path):  # noqa: ANN202 - test helper
    """Build the hard text corpus + pre-rendered mutations once for a test."""

    fingerprinter = Fingerprinter()
    index, fingerprints, texts = _hard_text_corpus(
        fingerprinter, np.random.default_rng(SEED), HARD_TEXT_CORPUS, tmp_path
    )
    mutations = dict(_hard_text_mutations(np.random.default_rng(SEED), texts))
    return fingerprinter, index, fingerprints, mutations


def test_hard_text_default_recall_drops_below_one(tmp_path: Path) -> None:
    # The whole point of the hard corpus: with the DEFAULT config the hardest
    # mutations are no longer saturated at 1.0, so a flag's recall win is now
    # measurable. scatter_insert and multi_point_edit fragment the offset
    # alignment; line_shuffle destroys it outright.
    fingerprinter, index, fingerprints, mutations = _hard_text(tmp_path)
    recall = {
        name: _mutation_recall(index, fingerprinter, tmp_path, fingerprints, render, ".txt")
        for name, render in mutations.items()
    }
    # Each of these must be discriminating (strictly below perfect recall).
    assert recall["scatter_insert_40"] < 1.0, recall
    assert recall["multi_point_edit"] < 1.0, recall
    assert recall["line_shuffle"] < 0.5, recall
    # And the corpus must not be SO hard that everything is zero -- a true match
    # is still usually findable, otherwise a flag could never show an improvement.
    assert recall["scatter_insert_40"] >= 0.5, recall


def test_hard_text_offset_tolerance_recovers_recall(tmp_path: Path) -> None:
    # offset_tolerance is a SEARCH-time flag: banding the winning offset bin over
    # +-tolerance adjacent delta bins re-gathers the votes a multi-edit fragmented,
    # recovering recall on the hardest multi_point_edit case WITHOUT regressing the
    # others and WITHOUT a false-accept cost on the plain corpus.
    fingerprinter, index, fingerprints, mutations = _hard_text(tmp_path)
    render = mutations["multi_point_edit"]
    off = _mutation_recall(index, fingerprinter, tmp_path, fingerprints, render, ".txt")
    on = _mutation_recall(
        index, fingerprinter, tmp_path, fingerprints, render, ".txt", offset_tolerance=2
    )
    assert on > off, f"offset_tolerance=2 did not recover recall: {off} -> {on}"
    # No precision cost from banding on the plain (non-sibling) corpus.
    assert _false_accept_rate(index, fingerprints, 0.05, offset_tolerance=2) == 0.0


def test_hard_text_generous_candidate_limit_is_lossless(tmp_path: Path) -> None:
    # candidate_limit is a cost prefilter, not a recall lever: a limit >= corpus
    # size selects every candidate, so recall must be byte-identical to full
    # search even on the hardest mutation. This pins the documented exactness.
    fingerprinter, index, fingerprints, mutations = _hard_text(tmp_path)
    render = mutations["scatter_insert_40"]
    full = _mutation_recall(index, fingerprinter, tmp_path, fingerprints, render, ".txt")
    generous = _mutation_recall(
        index, fingerprinter, tmp_path, fingerprints, render, ".txt", candidate_limit=HARD_TEXT_CORPUS
    )
    assert generous == full


def test_hard_confusability_precision_needs_stricter_cutoff(tmp_path: Path) -> None:
    # The sibling corpus creates genuine false-accept risk: at the default 0.05
    # cutoff impostors leak (precision loss is now MEASURABLE, not hidden), while
    # a stricter 0.20 cutoff cleans them up. This pins that the harness can
    # measure a precision/cutoff trade at all -- the data a "promote a
    # collision-adding flag to default" decision needs.
    rows = {r["flag"] + ":" + r["setting"]: r for r in confusability_precision(
        np.random.default_rng(SEED), families=4, siblings_per_family=3, workdir=tmp_path
    )}
    default = rows["default:off"]
    # Impostors sit AROUND the 0.05 cutoff -> leaks at 0.05, clean at 0.20.
    assert default["false_accept@0.05"] > 0.0, default
    assert default["false_accept@0.20"] == 0.0, default
    # freq_quantization adds collisions -> impostor confidence rises, not falls.
    assert rows["freq_quantization:4"]["mean_impostor"] >= default["mean_impostor"]


def test_audio_window_bank_recall_and_separation_on_hard_corpus(tmp_path: Path) -> None:
    pytest.importorskip("scipy", exc_type=ImportError)
    # The audio window bank (now the v2 DEFAULT) recalls hard clip/excerpt
    # mutations strongly with NO false accepts. NOTE: the v2 float64 reductions
    # fixed the float32-mean-cancellation that used to floor single-resolution
    # excerpt recall, so single-resolution is no longer at the floor on these
    # synthetic corpora -- the bank is the default for real non-stationary audio,
    # not to clear a synthetic floor. We therefore pin the bank's ABSOLUTE
    # quality (high recall, clean separation) rather than a delta over single.
    comparisons = {
        c.mutation: c for c in sweep_audio_flags(np.random.default_rng(SEED), HARD_AUDIO_CORPUS, tmp_path)
    }
    for name in ("clip_prefix_60pct", "excerpt_mid"):
        c = comparisons[name]
        assert c.on_recall >= 0.8, f"window_bank recall too low on {name}: {c.on_recall}"
        assert c.on_recall >= c.off_recall, f"bank regressed {name}: {c.off_recall} -> {c.on_recall}"
        assert c.on_false_accept == 0.0


def test_hard_image_phash_recovers_recall_at_precision_cost(tmp_path: Path) -> None:
    pytest.importorskip("PIL", exc_type=ImportError)
    # pHash recovers recall on the hard raster weak spots (strong resize, crop,
    # rotate, jpeg+crop) -- a REAL recall win, not just confidence. But on these
    # smooth synthetic images its global descriptor also raises impostor
    # confidence, so the default 0.05 cutoff leaks: pHash buys recall at a
    # precision cost that a stricter cutoff must absorb. Both halves are pinned.
    comparisons = {
        c.mutation: c for c in sweep_image_flags(np.random.default_rng(SEED), HARD_IMAGE_CORPUS, tmp_path)
    }
    # Recall win on the resize and jpeg+crop cases that raster collapses on.
    for name in ("resize_64x48", "jpeg_crop_resize_q25"):
        c = comparisons[name]
        assert c.on_recall > c.off_recall, f"phash did not lift {name}: {c.off_recall} -> {c.on_recall}"
    # The measurable precision cost: phash leaks false accepts at 0.05 here.
    assert comparisons["resize_64x48"].on_false_accept > 0.0
