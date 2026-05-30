"""Deterministic accuracy harness validating the engine's recall/calibration claims.

Unlike :mod:`benchmarks.benchmark` (which measures *speed* on whatever real
corpus it scans, and is therefore non-deterministic and slow), this harness
GENERATES a small, fully reproducible in-memory corpus and a MUTATION MATRIX so
the engine's matching *quality* can be asserted as a regression gate. It scans
nothing on disk beyond a private temp directory, is seeded end to end (numpy
``default_rng`` -- no wall-clock or unseeded ``random``), and runs in a couple
of seconds, so it is safe to call from a unit test and from CI.

It reports, per content type:

* exact self-match recall@1 (a fingerprint must find itself);
* near-duplicate recall@1 under each mutation -- text: front-insert, append,
  truncate/prefix, light and heavier char edits; image (if Pillow is present):
  downscale-resize and lossy JPEG re-encode; audio (if scipy is present): a
  prefix clip and a middle excerpt;
* a confidence-threshold SWEEP: precision, recall, and the false-accept rate
  (impostor accepted / impostor queries) at a ladder of thresholds, so the
  operating point implied by :class:`Calibration` is visible and tunable;
* the confidence SEPARATION: the true-match confidence distribution vs the
  impostor (best wrong file) distribution, which is the property a single
  ``default_min_confidence`` threshold relies on.

Run it standalone for the full human-readable sweep::

    python benchmarks/accuracy.py                 # JSON (default)
    python benchmarks/accuracy.py --format markdown
    python benchmarks/accuracy.py --text-corpus 40 --seed 7

The numbers it asserts in ``tests/test_accuracy.py`` are intentionally the
*documented, load-bearing* claims (exact recall == 1.0; text append/prefix
near-dup recall == 1.0; mean true-match confidence >= 0.5 while mean impostor
confidence < 0.05; pruning stays lossless). Audio clip/excerpt recall is
*measured and reported* but not asserted high: a whole-signal re-normalisation
shifts the audio time grid on a clip, which the current fixed-window matcher
does not survive -- the harness surfaces that honestly so the deferred matcher
research (window-bank / bin quantization) has a baseline to beat.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fingerprint_engine.core.fingerprinter import Fingerprinter
from fingerprint_engine.core.index import InMemoryHashIndex
from fingerprint_engine.core.models import Fingerprint

# A modest, deterministic vocabulary keeps generated text decodable as UTF-8 and
# rich enough that a fingerprint of one document is highly distinct from another.
_WORDS = (
    "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi "
    "omicron pi rho sigma tau upsilon phi chi psi omega quark lepton boson photon "
    "neutron proton electron muon gluon hadron baryon meson tensor scalar vector"
).split()

# Confidence-threshold ladder for the sweep. Spans the documented impostor band
# (< 0.05) through strong near-dup territory so the precision/recall/false-accept
# trade-off is legible. Kept as floats with a clean step for stable reporting.
_THRESHOLD_LADDER = (0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90)


@dataclass
class MutationResult:
    """recall@1 and confidence stats for one mutation applied across a corpus."""

    name: str
    queries: int
    hits: int
    mean_confidence: float
    min_confidence: float

    @property
    def recall_at_1(self) -> float:
        return self.hits / self.queries if self.queries else 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "mutation": self.name,
            "queries": self.queries,
            "hits": self.hits,
            "recall_at_1": round(self.recall_at_1, 4),
            "mean_confidence": round(self.mean_confidence, 4),
            "min_confidence": round(self.min_confidence, 4),
        }


@dataclass
class HandlerReport:
    """All accuracy measurements for one content type's generated corpus."""

    handler: str
    corpus_size: int
    avg_hashes_per_file: float
    exact_recall_at_1: float
    mean_true_confidence: float
    min_true_confidence: float
    mean_impostor_confidence: float
    max_impostor_confidence: float
    mutations: list[MutationResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "handler": self.handler,
            "corpus_size": self.corpus_size,
            "avg_hashes_per_file": round(self.avg_hashes_per_file, 1),
            "exact_recall_at_1": round(self.exact_recall_at_1, 4),
            "confidence_separation": {
                "mean_true": round(self.mean_true_confidence, 4),
                "min_true": round(self.min_true_confidence, 4),
                "mean_impostor": round(self.mean_impostor_confidence, 4),
                "max_impostor": round(self.max_impostor_confidence, 4),
                # The headline gap a single threshold relies on.
                "gap_mean": round(self.mean_true_confidence - self.mean_impostor_confidence, 4),
            },
            "mutations": [m.to_dict() for m in self.mutations],
        }


# ---------------------------------------------------------------------------
# Corpus generation (seeded; nothing read from disk but a private temp dir)
# ---------------------------------------------------------------------------


def _gen_document(rng: np.random.Generator, n_lines: int) -> str:
    """Build one pseudo-random text document of word-salad lines."""

    lines: list[str] = []
    for _ in range(n_lines):
        count = int(rng.integers(5, 13))
        idx = rng.integers(0, len(_WORDS), size=count)
        lines.append(" ".join(_WORDS[int(i)] for i in idx))
    return "\n".join(lines) + "\n"


def _write_text_corpus(rng: np.random.Generator, size: int, workdir: Path) -> tuple[list[str], list[str]]:
    """Write ``size`` distinct text files; return their (paths, source texts)."""

    paths: list[str] = []
    texts: list[str] = []
    for i in range(size):
        text = _gen_document(rng, n_lines=int(rng.integers(60, 100)))
        path = workdir / f"doc{i:03d}.txt"
        path.write_text(text, encoding="utf-8")
        paths.append(str(path))
        texts.append(text)
    return paths, texts


def _gen_image(rng: np.random.Generator, width: int, height: int):  # noqa: ANN202 - PIL optional
    """Build one RGB image: smooth gradients (survive resize) + per-file noise."""

    from PIL import Image

    arr = rng.integers(0, 256, size=(height, width, 3)).astype(np.uint8)
    gx = np.linspace(0, 255, width).astype(np.uint8)
    gy = np.linspace(0, 255, height).astype(np.uint8)
    arr[:, :, 0] = gx[None, :]
    arr[:, :, 1] = gy[:, None]
    return Image.fromarray(arr, "RGB")


def _write_image_corpus(rng: np.random.Generator, size: int, workdir: Path):  # noqa: ANN202
    """Write ``size`` PNGs; return (paths, PIL images) for later mutation."""

    paths: list[str] = []
    images = []
    for i in range(size):
        width = int(rng.integers(140, 200))
        height = int(rng.integers(110, 160))
        image = _gen_image(rng, width, height)
        path = workdir / f"img{i:03d}.png"
        image.save(path)
        paths.append(str(path))
        images.append(image)
    return paths, images


def _gen_wave(rng: np.random.Generator, sample_rate: int, seconds: float) -> tuple[int, np.ndarray]:
    """Build one mono 16-bit WAV: a distinct 3-tone chord plus light noise."""

    n = int(sample_rate * seconds)
    t = np.arange(n) / sample_rate
    f1 = float(rng.integers(200, 800))
    f2 = float(rng.integers(900, 2000))
    f3 = float(rng.integers(2100, 3500))
    signal = (
        0.5 * np.sin(2 * np.pi * f1 * t)
        + 0.3 * np.sin(2 * np.pi * f2 * t)
        + 0.2 * np.sin(2 * np.pi * f3 * t)
        + 0.03 * rng.standard_normal(n)
    )
    peak = float(np.max(np.abs(signal))) or 1.0
    return sample_rate, (signal / peak * 32767).astype(np.int16)


def _write_audio_corpus(
    rng: np.random.Generator, size: int, workdir: Path
) -> tuple[list[str], list[tuple[int, np.ndarray]]]:
    """Write ``size`` WAVs; return (paths, (sample_rate, samples) tuples)."""

    from scipy.io import wavfile

    paths: list[str] = []
    waves: list[tuple[int, np.ndarray]] = []
    for i in range(size):
        sample_rate, data = _gen_wave(rng, sample_rate=8000, seconds=3.0)
        path = workdir / f"aud{i:03d}.wav"
        wavfile.write(str(path), sample_rate, data)
        paths.append(str(path))
        waves.append((sample_rate, data))
    return paths, waves


# ---------------------------------------------------------------------------
# Measurement primitives
# ---------------------------------------------------------------------------


def _exact_recall_and_separation(
    index: InMemoryHashIndex, fingerprints: list[Fingerprint]
) -> tuple[float, list[float], list[float]]:
    """Return (exact recall@1, true-match confidences, best-impostor confidences).

    Each indexed fingerprint is searched against the index it lives in. ``recall``
    is the share whose top-1 is itself; ``true`` collects the self-match
    confidence; ``impostor`` collects the best confidence among *other* files
    (the hardest wrong answer), which is what a threshold must reject.
    """

    hits = 0
    true_conf: list[float] = []
    impostor_conf: list[float] = []
    for fingerprint in fingerprints:
        results = index.search(fingerprint, top_k=5)
        if results and results[0].file_id == fingerprint.file_id:
            hits += 1
        true_conf.append(
            next((r.confidence for r in results if r.file_id == fingerprint.file_id), 0.0)
        )
        impostor_conf.append(
            max((r.confidence for r in results if r.file_id != fingerprint.file_id), default=0.0)
        )
    recall = hits / len(fingerprints) if fingerprints else 0.0
    return recall, true_conf, impostor_conf


def _measure_mutation(
    name: str,
    index: InMemoryHashIndex,
    fingerprinter: Fingerprinter,
    workdir: Path,
    targets: list[Fingerprint],
    render: Callable[[int], bytes],
    suffix: str,
) -> MutationResult:
    """Apply a mutation to each target, re-fingerprint, and score recall@1.

    ``render(i)`` returns the mutated bytes for the i-th target; the bytes are
    written to a scratch file with ``suffix`` (so the right handler routes) and
    searched. A hit is a top-1 match back to the original file_id.
    """

    hits = 0
    confidences: list[float] = []
    scratch = workdir / f"_mut_{name}{suffix}"
    for i, target in enumerate(targets):
        scratch.write_bytes(render(i))
        results = index.search(fingerprinter.fingerprint_file(scratch), top_k=1)
        if results and results[0].file_id == target.file_id:
            hits += 1
            confidences.append(results[0].confidence)
    mean_conf = statistics.mean(confidences) if confidences else 0.0
    min_conf = min(confidences) if confidences else 0.0
    return MutationResult(
        name=name, queries=len(targets), hits=hits, mean_confidence=mean_conf, min_confidence=min_conf
    )


def _char_edit(rng: np.random.Generator, text: str, fraction: float) -> str:
    """Flip ``fraction`` of code points to a neighbouring printable character."""

    chars = list(text)
    n = len(chars)
    if n == 0:
        return text
    count = max(1, int(n * fraction))
    positions = rng.choice(n, size=count, replace=False)
    for pos in positions:
        code = ord(chars[int(pos)])
        chars[int(pos)] = chr(33 + (code + 1) % 90)  # stay in printable ASCII
    return "".join(chars)


# ---------------------------------------------------------------------------
# Per-handler evaluation
# ---------------------------------------------------------------------------


def evaluate_text(
    fingerprinter: Fingerprinter, rng: np.random.Generator, size: int, workdir: Path
) -> tuple[HandlerReport, InMemoryHashIndex, list[Fingerprint]]:
    """Build a text corpus, index it, and measure exact + near-dup accuracy."""

    paths, texts = _write_text_corpus(rng, size, workdir)
    fingerprints = [fingerprinter.fingerprint_file(p) for p in paths]
    index = InMemoryHashIndex()
    index.add_many(fingerprints)

    recall, true_conf, impostor_conf = _exact_recall_and_separation(index, fingerprints)

    # Pre-render deterministic mutation payloads so each is a pure function of
    # the seed (the scratch-file rewrite inside _measure_mutation is the only
    # side effect).
    insert_block = _gen_document(rng, n_lines=6)
    edits_light = [_char_edit(rng, texts[i], 0.02) for i in range(size)]
    edits_heavy = [_char_edit(rng, texts[i], 0.05) for i in range(size)]

    mutations = [
        _measure_mutation(
            "front_insert", index, fingerprinter, workdir, fingerprints,
            lambda i: (insert_block + texts[i]).encode("utf-8"), ".txt",
        ),
        _measure_mutation(
            "append", index, fingerprinter, workdir, fingerprints,
            lambda i: (texts[i] + "# appended trailer line\n" * 5).encode("utf-8"), ".txt",
        ),
        _measure_mutation(
            "truncate_prefix", index, fingerprinter, workdir, fingerprints,
            lambda i: texts[i][: int(len(texts[i]) * 0.6)].encode("utf-8"), ".txt",
        ),
        _measure_mutation(
            "char_edit_2pct", index, fingerprinter, workdir, fingerprints,
            lambda i: edits_light[i].encode("utf-8"), ".txt",
        ),
        _measure_mutation(
            "char_edit_5pct", index, fingerprinter, workdir, fingerprints,
            lambda i: edits_heavy[i].encode("utf-8"), ".txt",
        ),
    ]

    report = _build_report("text", fingerprints, recall, true_conf, impostor_conf, mutations)
    return report, index, fingerprints


def evaluate_image(
    fingerprinter: Fingerprinter, rng: np.random.Generator, size: int, workdir: Path
) -> HandlerReport:
    """Build an image corpus and measure exact + resize/JPEG near-dup recall."""

    from PIL import Image  # noqa: F401 - import gates the whole case on Pillow

    paths, images = _write_image_corpus(rng, size, workdir)
    fingerprints = [fingerprinter.fingerprint_file(p) for p in paths]
    index = InMemoryHashIndex()
    index.add_many(fingerprints)

    recall, true_conf, impostor_conf = _exact_recall_and_separation(index, fingerprints)

    import io

    def _resized(i: int) -> bytes:
        buffer = io.BytesIO()
        images[i].resize((100, 75)).save(buffer, format="PNG")
        return buffer.getvalue()

    def _jpeg(i: int) -> bytes:
        buffer = io.BytesIO()
        images[i].save(buffer, format="JPEG", quality=40)
        return buffer.getvalue()

    def _crop(i: int) -> bytes:
        # Trim a 10% border off every edge. A raster row-major signal is wrecked
        # by this (every row shifts after the canonical resize); a DCT pHash, a
        # low-frequency global descriptor, is far more tolerant. Disclaimed as a
        # raster limitation in the README -- measured here for the phash compare.
        image = images[i]
        width, height = image.size
        box = (int(width * 0.1), int(height * 0.1), int(width * 0.9), int(height * 0.9))
        buffer = io.BytesIO()
        image.crop(box).save(buffer, format="PNG")
        return buffer.getvalue()

    def _rotate(i: int) -> bytes:
        # A small 5-degree rotation (expand so nothing is clipped). Same story as
        # crop: catastrophic for the raster row signal, gentler on the pHash.
        buffer = io.BytesIO()
        images[i].rotate(5, expand=True, fillcolor=128).save(buffer, format="PNG")
        return buffer.getvalue()

    mutations = [
        _measure_mutation("resize_downscale", index, fingerprinter, workdir, fingerprints, _resized, ".png"),
        _measure_mutation("jpeg_q40", index, fingerprinter, workdir, fingerprints, _jpeg, ".jpg"),
        _measure_mutation("crop_border_10pct", index, fingerprinter, workdir, fingerprints, _crop, ".png"),
        _measure_mutation("rotate_5deg", index, fingerprinter, workdir, fingerprints, _rotate, ".png"),
    ]
    return _build_report("image", fingerprints, recall, true_conf, impostor_conf, mutations)


def evaluate_audio(
    fingerprinter: Fingerprinter, rng: np.random.Generator, size: int, workdir: Path
) -> HandlerReport:
    """Build an audio corpus and measure exact recall + clip/excerpt recall.

    Clip/excerpt recall is expected to be LOW with the current matcher (a clip
    re-normalises the whole signal and shifts the fixed-window time grid). It is
    reported, not asserted, as a baseline for the deferred matcher research.
    """

    from scipy.io import wavfile

    paths, waves = _write_audio_corpus(rng, size, workdir)
    fingerprints = [fingerprinter.fingerprint_file(p) for p in paths]
    index = InMemoryHashIndex()
    index.add_many(fingerprints)

    recall, true_conf, impostor_conf = _exact_recall_and_separation(index, fingerprints)

    def _clip(fraction_start: float, fraction_end: float) -> Callable[[int], bytes]:
        def render(i: int) -> bytes:
            import io

            sample_rate, data = waves[i]
            start = int(len(data) * fraction_start)
            end = int(len(data) * fraction_end)
            buffer = io.BytesIO()
            wavfile.write(buffer, sample_rate, data[start:end])
            return buffer.getvalue()

        return render

    mutations = [
        _measure_mutation("clip_prefix_60pct", index, fingerprinter, workdir, fingerprints, _clip(0.0, 0.6), ".wav"),
        _measure_mutation("excerpt_mid", index, fingerprinter, workdir, fingerprints, _clip(0.2, 0.8), ".wav"),
    ]
    return _build_report("audio", fingerprints, recall, true_conf, impostor_conf, mutations)


def _build_report(
    handler: str,
    fingerprints: list[Fingerprint],
    recall: float,
    true_conf: list[float],
    impostor_conf: list[float],
    mutations: list[MutationResult],
) -> HandlerReport:
    avg_hashes = (sum(f.hash_count for f in fingerprints) / len(fingerprints)) if fingerprints else 0.0
    return HandlerReport(
        handler=handler,
        corpus_size=len(fingerprints),
        avg_hashes_per_file=avg_hashes,
        exact_recall_at_1=recall,
        mean_true_confidence=statistics.mean(true_conf) if true_conf else 0.0,
        min_true_confidence=min(true_conf) if true_conf else 0.0,
        mean_impostor_confidence=statistics.mean(impostor_conf) if impostor_conf else 0.0,
        max_impostor_confidence=max(impostor_conf) if impostor_conf else 0.0,
        mutations=mutations,
    )


# ---------------------------------------------------------------------------
# Confidence-threshold sweep (precision / recall / false-accept rate)
# ---------------------------------------------------------------------------


def threshold_sweep(
    index: InMemoryHashIndex,
    fingerprints: list[Fingerprint],
    thresholds: tuple[float, ...] = _THRESHOLD_LADDER,
) -> list[dict[str, object]]:
    """Sweep ``Calibration.default_min_confidence`` and score the operating point.

    For each threshold we run two query populations against the SAME index:

    * GENUINE queries -- every indexed fingerprint searched against itself. The
      top-1 (if it survives calibration) should be the file itself.
    * IMPOSTOR queries -- each fingerprint searched against an index that has had
      that file removed, so *no* correct answer exists; any accepted top-1 is a
      false accept.

    From those we derive, at each threshold: recall (genuine accepted as self),
    precision (true accepts / all accepts), and false_accept_rate (impostor
    queries that returned any accepted result). This makes the recall-vs-false-
    accept trade-off -- and the basis for the default 0.05 cutoff -- explicit.
    """

    rows: list[dict[str, object]] = []
    n = len(fingerprints)
    if n == 0:
        return rows

    # Precompute, once, the best genuine top-1 confidence per query and the best
    # impostor top-1 confidence per query, so the sweep is just thresholding.
    genuine_top: list[tuple[bool, float]] = []  # (is_self, confidence) of top-1
    for fingerprint in fingerprints:
        results = index.search(fingerprint, top_k=1)
        if results:
            genuine_top.append((results[0].file_id == fingerprint.file_id, results[0].confidence))
        else:
            genuine_top.append((False, 0.0))

    # Impostor population: drop each file, search, restore. Leave-one-out keeps
    # the index size comparable and guarantees no correct answer is reachable.
    impostor_top: list[float] = []
    for fingerprint in fingerprints:
        index.remove(fingerprint.file_id)
        results = index.search(fingerprint, top_k=1)
        impostor_top.append(results[0].confidence if results else 0.0)
        index.add(fingerprint)

    for threshold in thresholds:
        true_accepts = sum(1 for is_self, conf in genuine_top if is_self and conf >= threshold)
        wrong_accepts = sum(1 for is_self, conf in genuine_top if (not is_self) and conf >= threshold)
        false_accepts = sum(1 for conf in impostor_top if conf >= threshold)
        total_accepts = true_accepts + wrong_accepts
        rows.append(
            {
                "threshold": round(threshold, 4),
                "recall": round(true_accepts / n, 4),
                "precision": round(true_accepts / total_accepts, 4) if total_accepts else 1.0,
                "false_accept_rate": round(false_accepts / n, 4),
            }
        )
    return rows


def prune_is_lossless(index: InMemoryHashIndex, fingerprints: list[Fingerprint]) -> dict[str, object]:
    """Prune stop-hashes and confirm exact self-match recall@1 is unchanged.

    Operates on a fresh copy (rebuilt from the same fingerprints) so the caller's
    index is untouched. Returns the before/after recall and postings removed.
    """

    pruned = InMemoryHashIndex()
    pruned.add_many(fingerprints)
    before, _, _ = _exact_recall_and_separation(pruned, fingerprints)
    removed = pruned.prune_stop_hashes()
    after, _, _ = _exact_recall_and_separation(pruned, fingerprints)
    return {
        "postings_removed": removed,
        "recall_before": round(before, 4),
        "recall_after": round(after, 4),
        "lossless": before == after == 1.0,
    }


# ---------------------------------------------------------------------------
# Top-level harness + reporting
# ---------------------------------------------------------------------------


def run_accuracy(
    *,
    seed: int = 1234,
    text_corpus: int = 36,
    image_corpus: int = 18,
    audio_corpus: int = 18,
    include_image: bool = True,
    include_audio: bool = True,
) -> dict[str, object]:
    """Run the full accuracy harness and return a structured report dict.

    Deterministic for a fixed ``seed``: a single ``default_rng(seed)`` drives all
    corpus and mutation generation, in a fixed call order, so two runs with the
    same seed produce byte-identical numbers. Image/audio sections are skipped
    (and recorded as skipped) when their optional dependency is unavailable.
    """

    rng = np.random.default_rng(seed)
    fingerprinter = Fingerprinter()
    report: dict[str, object] = {"seed": seed, "handlers": {}, "sweep": {}, "skipped": []}
    handlers: dict[str, object] = report["handlers"]  # type: ignore[assignment]
    sweeps: dict[str, object] = report["sweep"]  # type: ignore[assignment]
    skipped: list[str] = report["skipped"]  # type: ignore[assignment]

    with tempfile.TemporaryDirectory(prefix="fp_accuracy_") as tmp:
        workdir = Path(tmp)

        text_report, text_index, text_fps = evaluate_text(fingerprinter, rng, text_corpus, workdir)
        handlers["text"] = text_report.to_dict()
        sweeps["text"] = threshold_sweep(text_index, text_fps)
        handlers["text"]["prune"] = prune_is_lossless(text_index, text_fps)  # type: ignore[index]

        if include_image:
            try:
                import PIL  # noqa: F401
            except ImportError:
                skipped.append("image (Pillow not installed)")
            else:
                handlers["image"] = evaluate_image(fingerprinter, rng, image_corpus, workdir).to_dict()

        if include_audio:
            try:
                import scipy  # noqa: F401
            except ImportError:
                skipped.append("audio (scipy not installed)")
            else:
                handlers["audio"] = evaluate_audio(fingerprinter, rng, audio_corpus, workdir).to_dict()

    return report


def _render_markdown(report: dict[str, object]) -> str:
    """Render the report as a human-readable markdown document."""

    lines: list[str] = [f"# Fingerprint engine accuracy report (seed={report['seed']})", ""]
    handlers: dict[str, dict] = report["handlers"]  # type: ignore[assignment]
    for name, data in handlers.items():
        sep = data["confidence_separation"]
        lines.append(f"## {name}  (corpus={data['corpus_size']}, avg hashes/file={data['avg_hashes_per_file']})")
        lines.append("")
        lines.append(f"- exact recall@1: **{data['exact_recall_at_1']}**")
        lines.append(
            f"- confidence separation: true mean **{sep['mean_true']}** (min {sep['min_true']}) "
            f"vs impostor mean **{sep['mean_impostor']}** (max {sep['max_impostor']}), "
            f"gap {sep['gap_mean']}"
        )
        if "prune" in data:
            prune = data["prune"]
            lines.append(
                f"- prune_stop_hashes: removed {prune['postings_removed']} postings, "
                f"recall {prune['recall_before']} -> {prune['recall_after']} "
                f"(lossless={prune['lossless']})"
            )
        lines.append("")
        lines.append("| mutation | recall@1 | mean conf | min conf | hits/queries |")
        lines.append("| --- | --- | --- | --- | --- |")
        for mutation in data["mutations"]:
            lines.append(
                f"| {mutation['mutation']} | {mutation['recall_at_1']} | {mutation['mean_confidence']} "
                f"| {mutation['min_confidence']} | {mutation['hits']}/{mutation['queries']} |"
            )
        lines.append("")

    sweeps: dict[str, list[dict]] = report["sweep"]  # type: ignore[assignment]
    for name, rows in sweeps.items():
        lines.append(f"## {name}: confidence-threshold sweep")
        lines.append("")
        lines.append("| threshold | recall | precision | false-accept rate |")
        lines.append("| --- | --- | --- | --- |")
        for row in rows:
            lines.append(
                f"| {row['threshold']} | {row['recall']} | {row['precision']} | {row['false_accept_rate']} |"
            )
        lines.append("")

    if report["skipped"]:
        lines.append("## skipped")
        lines.append("")
        for item in report["skipped"]:  # type: ignore[union-attr]
            lines.append(f"- {item}")
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=1234, help="RNG seed (deterministic output)")
    parser.add_argument("--text-corpus", type=int, default=36, help="number of text files to generate")
    parser.add_argument("--image-corpus", type=int, default=18, help="number of images to generate")
    parser.add_argument("--audio-corpus", type=int, default=18, help="number of audio clips to generate")
    parser.add_argument("--no-image", action="store_true", help="skip the image section even if Pillow is present")
    parser.add_argument("--no-audio", action="store_true", help="skip the audio section even if scipy is present")
    parser.add_argument("--format", choices=("json", "markdown"), default="json", help="output format")
    args = parser.parse_args(argv)

    report = run_accuracy(
        seed=args.seed,
        text_corpus=args.text_corpus,
        image_corpus=args.image_corpus,
        audio_corpus=args.audio_corpus,
        include_image=not args.no_image,
        include_audio=not args.no_audio,
    )

    if args.format == "markdown":
        print(_render_markdown(report))
    else:
        print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
