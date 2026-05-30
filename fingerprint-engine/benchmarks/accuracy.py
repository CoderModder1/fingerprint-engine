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
from typing import Literal

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fingerprint_engine.core.fingerprinter import Fingerprinter
from fingerprint_engine.core.index import InMemoryHashIndex
from fingerprint_engine.core.models import Fingerprint, FingerprintConfig

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
# HARD evaluation mode
#
# The default near-dup matrix above is deliberately gentle: it documents the
# happy path and, by design, every text/image case there recalls at 1.0 even
# under heavy char edits (the offset histogram is extremely robust -- the large
# UNTOUCHED spans of a doc all vote coherently at offset 0, so a true match wins
# however much you scribble on it). That saturation is the BLIND SPOT: an opt-in
# matching flag that genuinely lifts recall is invisible when recall is already
# 1.0, so a flag's win can only show up as confidence, never as recall.
#
# This section builds a HARDER corpus + mutation matrix whose explicit purpose is
# to push default-config recall@1 BELOW 1.0, so a flag's effect on RECALL (not
# just confidence) is measurable, and to inject genuine false-accept pressure so
# PRECISION is measurable too. Two pressures combine:
#
# * AGGRESSIVE mutations that fragment the offset alignment -- many scattered
#   insertions (each shifts the absolute frame index of everything after it, so
#   aligned votes smear across a wide band of delta bins instead of piling on one
#   bin), heavy multi-point char edits, line shuffles; for image, strong
#   downscale + crop + rotate + low-quality JPEG; for audio, the excerpt clip
#   that re-normalises the global time grid.
# * CONFUSABILITY pressure -- "sibling" documents that share a common header
#   skeleton but carry distinct payload bodies. Siblings genuinely collide on the
#   shared region, so an impostor's best wrong-file confidence rises toward the
#   0.05 cutoff and a real false-accept risk exists to measure. The shared
#   fraction is tuned so impostors sit AROUND the cutoff (not far above it): the
#   default operating point leaks some false accepts, and a stricter cutoff
#   recovers precision -- which is the trade a "promote a collision-adding flag to
#   default" decision must weigh.
#
# Everything here is seeded off the same ``default_rng`` and bounded to small
# corpora so it stays a few-seconds unit-testable harness, never a benchmark run.
# ---------------------------------------------------------------------------

# Each hard text mutation is keyed by name -> (description, builder). A builder is
# ``(rng, text) -> str``; recall is read OFF (default config) vs ON (each flag).
HARD_SCATTER_INSERTS = 40  # scattered single-line inserts: enough to fragment
HARD_CHAR_EDIT_FRACTION = 0.20  # heavy multi-point edits (20% of code points)


def _scatter_insert(rng: np.random.Generator, text: str, n_inserts: int) -> str:
    """Splice ``n_inserts`` random word-salad lines at scattered positions.

    Each insertion shifts the frame index of all following content by a little,
    so the true match's aligned votes fragment across a wide band of delta bins
    instead of concentrating on the single offset-0 bin -- the exact failure mode
    the offset histogram is weakest against and that drops default recall below
    1.0 once enough insertions accumulate.
    """

    lines = text.split("\n")
    for _ in range(n_inserts):
        pos = int(rng.integers(0, len(lines) + 1))
        count = int(rng.integers(5, 13))
        idx = rng.integers(0, len(_WORDS), size=count)
        lines.insert(pos, " ".join(_WORDS[int(i)] for i in idx))
    return "\n".join(lines) + "\n"


def _line_shuffle(rng: np.random.Generator, text: str) -> str:
    """Permute the lines of a document (destroys all sequential structure)."""

    lines = text.split("\n")
    perm = rng.permutation(len(lines))
    return "\n".join(lines[int(i)] for i in perm) + "\n"


def _multi_point_edit(rng: np.random.Generator, text: str, n_inserts: int, fraction: float) -> str:
    """Combine scattered inserts with a heavy char-edit pass (the worst case)."""

    return _char_edit(rng, _scatter_insert(rng, text, n_inserts), fraction)


# Number of shared header lines vs distinct payload lines in a sibling. Tuned
# (see benchmarks/RESULTS.md hard-mode notes) so a sibling's best wrong-match
# confidence lands AROUND the 0.05 cutoff: enough collision to leak false accepts
# at 0.05 (measurable precision loss), little enough that a stricter cutoff cleans
# it up -- the band where a precision/recall trade is actually visible.
HARD_SIBLING_SHARED_LINES = 12
HARD_SIBLING_PAYLOAD_LINES = 45


def _gen_sibling_lines(rng: np.random.Generator, n_lines: int) -> list[str]:
    """Build ``n_lines`` of word-salad as a list (so headers/payloads compose)."""

    lines: list[str] = []
    for _ in range(n_lines):
        count = int(rng.integers(5, 13))
        idx = rng.integers(0, len(_WORDS), size=count)
        lines.append(" ".join(_WORDS[int(i)] for i in idx))
    return lines


def _write_sibling_corpus(
    rng: np.random.Generator, families: int, siblings_per_family: int, workdir: Path
) -> tuple[list[str], list[str]]:
    """Write a confusability corpus of sibling documents.

    Each of ``families`` families shares one randomly generated header skeleton;
    every sibling in a family appends its own distinct payload body. Siblings
    therefore collide on the shared header (the false-accept pressure) while
    remaining distinct documents (the payload dominates the fingerprint). Returns
    (paths, source texts) in a fixed order so the corpus is fully reproducible.
    """

    paths: list[str] = []
    texts: list[str] = []
    doc = 0
    for _ in range(families):
        header = _gen_sibling_lines(rng, HARD_SIBLING_SHARED_LINES)
        for _ in range(siblings_per_family):
            payload = _gen_sibling_lines(rng, HARD_SIBLING_PAYLOAD_LINES)
            text = "\n".join(header + payload) + "\n"
            path = workdir / f"sib{doc:03d}.txt"
            path.write_text(text, encoding="utf-8")
            paths.append(str(path))
            texts.append(text)
            doc += 1
    return paths, texts


def _gen_hard_image(rng: np.random.Generator, width: int, height: int):  # noqa: ANN202 - PIL optional
    """Build a SMOOTH, per-file-distinct image for the hard image sweep.

    The standard harness's gradient image is identical across files apart from
    additive noise, so a global pHash descriptor (a low-frequency summary) makes
    every file look alike and impostor confidence is uninformative. Here each file
    is a sum of a few RANDOM low-frequency 2D sinusoids: smooth (so a pHash
    survives crop/rotate/resize -- a real recall signal) yet per-file-distinct in
    its global structure (so impostor pHashes separate and the precision cost of
    pHash is measured honestly, not masked by a shared gradient).
    """

    from PIL import Image

    yy, xx = np.mgrid[0:height, 0:width].astype(float)
    field = np.zeros((height, width))
    for _ in range(4):
        fx = rng.uniform(0.5, 3.0) / width
        fy = rng.uniform(0.5, 3.0) / height
        phase = rng.uniform(0, 2 * np.pi)
        amp = rng.uniform(0.5, 1.0)
        field += amp * np.sin(2 * np.pi * (fx * xx + fy * yy) + phase)
    field -= field.min()
    field /= field.max() or 1.0
    arr = np.empty((height, width, 3), dtype=np.uint8)
    arr[:, :, 0] = (field * 255).astype(np.uint8)
    arr[:, :, 1] = ((1.0 - field) * 255).astype(np.uint8)
    arr[:, :, 2] = (field * field * 255).astype(np.uint8)
    return Image.fromarray(arr, "RGB")


def _write_hard_image_corpus(rng: np.random.Generator, size: int, workdir: Path):  # noqa: ANN202
    """Write ``size`` smooth per-file-distinct PNGs; return (paths, PIL images)."""

    paths: list[str] = []
    images = []
    for i in range(size):
        width = int(rng.integers(140, 200))
        height = int(rng.integers(110, 160))
        image = _gen_hard_image(rng, width, height)
        path = workdir / f"himg{i:03d}.png"
        image.save(path)
        paths.append(str(path))
        images.append(image)
    return paths, images


# ---------------------------------------------------------------------------
# Per-flag OFF-vs-ON measurement on the hard corpus
# ---------------------------------------------------------------------------


@dataclass
class FlagComparison:
    """recall@1 and precision/false-accept for one mutation, OFF vs ON a flag.

    ``*_false_accept`` is the leave-one-out false-accept rate at the default 0.05
    cutoff; ``*_false_accept_strict`` is the same at a stricter 0.20 cutoff. A
    flag whose recall win comes with a high 0.05 false-accept but a clean 0.20 one
    is justifiable only at the stricter operating point -- the table makes that
    explicit so a "promote to default" decision sees the cutoff it would require.
    """

    flag: str
    setting: str
    mutation: str
    off_recall: float
    on_recall: float
    off_false_accept: float
    on_false_accept: float
    off_false_accept_strict: float = 0.0
    on_false_accept_strict: float = 0.0

    @property
    def recall_delta(self) -> float:
        return self.on_recall - self.off_recall

    def to_dict(self) -> dict[str, object]:
        return {
            "flag": self.flag,
            "setting": self.setting,
            "mutation": self.mutation,
            "off_recall": round(self.off_recall, 4),
            "on_recall": round(self.on_recall, 4),
            "recall_delta": round(self.recall_delta, 4),
            "off_false_accept@0.05": round(self.off_false_accept, 4),
            "on_false_accept@0.05": round(self.on_false_accept, 4),
            "off_false_accept@0.20": round(self.off_false_accept_strict, 4),
            "on_false_accept@0.20": round(self.on_false_accept_strict, 4),
        }


def _false_accept_rate(
    index: InMemoryHashIndex,
    fingerprints: list[Fingerprint],
    cutoff: float,
    *,
    offset_tolerance: int | None = None,
    candidate_limit: int | None = None,
) -> float:
    """Leave-one-out false-accept rate at ``cutoff``.

    Each fingerprint is removed from the index, searched (so no correct answer is
    reachable), and restored; any returned top-1 whose confidence clears
    ``cutoff`` is a false accept. The index is left exactly as it was found.
    """

    if not fingerprints:
        return 0.0
    false_accepts = 0
    for fingerprint in fingerprints:
        index.remove(fingerprint.file_id)
        results = index.search(
            fingerprint, top_k=1, offset_tolerance=offset_tolerance, candidate_limit=candidate_limit
        )
        if results and results[0].confidence >= cutoff:
            false_accepts += 1
        index.add(fingerprint)
    return false_accepts / len(fingerprints)


def _mutation_recall(
    index: InMemoryHashIndex,
    fingerprinter: Fingerprinter,
    workdir: Path,
    targets: list[Fingerprint],
    render: Callable[[int], bytes],
    suffix: str,
    *,
    offset_tolerance: int | None = None,
    candidate_limit: int | None = None,
) -> float:
    """recall@1 of a mutation, with optional search-time flags applied."""

    if not targets:
        return 0.0
    hits = 0
    scratch = workdir / f"_hard_mut{suffix}"
    for i, target in enumerate(targets):
        scratch.write_bytes(render(i))
        query = fingerprinter.fingerprint_file(scratch)
        results = index.search(
            query, top_k=1, offset_tolerance=offset_tolerance, candidate_limit=candidate_limit
        )
        if results and results[0].file_id == target.file_id:
            hits += 1
    return hits / len(targets)


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
# Hard-corpus per-flag sweeps. Each returns a list[FlagComparison]: the OFF
# (default-config) recall and false-accept vs the ON recall and false-accept,
# per hard mutation. The corpus and mutations are rebuilt identically (same seed,
# same fixed RNG call order) for OFF and ON so the only variable is the flag.
# ---------------------------------------------------------------------------


def _hard_text_corpus(
    fingerprinter: Fingerprinter, rng: np.random.Generator, size: int, workdir: Path
) -> tuple[InMemoryHashIndex, list[Fingerprint], list[str]]:
    """Index a plain text corpus and return (index, fingerprints, source texts)."""

    paths, texts = _write_text_corpus(rng, size, workdir)
    fingerprints = [fingerprinter.fingerprint_file(p) for p in paths]
    index = InMemoryHashIndex()
    index.add_many(fingerprints)
    return index, fingerprints, texts


def _hard_text_mutations(
    rng: np.random.Generator, texts: list[str]
) -> list[tuple[str, Callable[[int], bytes]]]:
    """Pre-render the deterministic hard text mutation payloads (pure of seed)."""

    scatter = [_scatter_insert(rng, t, HARD_SCATTER_INSERTS) for t in texts]
    shuffle = [_line_shuffle(rng, t) for t in texts]
    heavy = [_char_edit(rng, t, HARD_CHAR_EDIT_FRACTION) for t in texts]
    multi = [_multi_point_edit(rng, t, HARD_SCATTER_INSERTS, HARD_CHAR_EDIT_FRACTION) for t in texts]
    return [
        ("scatter_insert_40", lambda i: scatter[i].encode("utf-8")),
        ("line_shuffle", lambda i: shuffle[i].encode("utf-8")),
        ("char_edit_20pct", lambda i: heavy[i].encode("utf-8")),
        ("multi_point_edit", lambda i: multi[i].encode("utf-8")),
    ]


def sweep_text_flags(
    rng: np.random.Generator, size: int, workdir: Path
) -> list[FlagComparison]:
    """OFF-vs-ON sweep of the text-relevant flags on the hard text corpus.

    Covers ``freq_quantization`` in {2, 4}, ``offset_tolerance`` in {1, 2}, and
    ``candidate_limit`` in {size, 5}. Each ON config rebuilds the corpus with the
    same seed so OFF/ON differ only by the flag. Reports recall@1 per hard
    mutation and the leave-one-out false-accept rate at the 0.05 cutoff.
    """

    # OFF baseline: default config, exact search. Built once and reused.
    off_index, off_fps, off_texts = _hard_text_corpus(Fingerprinter(), rng, size, workdir)
    off_mutations = _hard_text_mutations(rng, off_texts)
    off_recall = {
        name: _mutation_recall(off_index, Fingerprinter(), workdir, off_fps, render, ".txt")
        for name, render in off_mutations
    }
    off_fa = _false_accept_rate(off_index, off_fps, 0.05)
    off_fa_strict = _false_accept_rate(off_index, off_fps, 0.20)

    comparisons: list[FlagComparison] = []

    def _record(
        flag: str, setting: str, on_recall: dict[str, float], on_fa: float, on_fa_strict: float
    ) -> None:
        for name in off_recall:
            comparisons.append(
                FlagComparison(
                    flag=flag,
                    setting=setting,
                    mutation=name,
                    off_recall=off_recall[name],
                    on_recall=on_recall[name],
                    off_false_accept=off_fa,
                    on_false_accept=on_fa,
                    off_false_accept_strict=off_fa_strict,
                    on_false_accept_strict=on_fa_strict,
                )
            )

    # freq_quantization: a hash-changing flag, so rebuild the corpus at each q.
    for q in (2, 4):
        cfg = FingerprintConfig(freq_quantization=q)
        fp = Fingerprinter(config=cfg)
        idx, fps, texts = _hard_text_corpus(fp, np.random.default_rng(_SEED_FOR(rng)), size, workdir)
        mutations = _hard_text_mutations(np.random.default_rng(_SEED_FOR(rng)), texts)
        on_recall = {
            name: _mutation_recall(idx, fp, workdir, fps, render, ".txt")
            for name, render in mutations
        }
        _record(
            "freq_quantization",
            str(q),
            on_recall,
            _false_accept_rate(idx, fps, 0.05),
            _false_accept_rate(idx, fps, 0.20),
        )

    # offset_tolerance: a SEARCH-time flag -- reuse the OFF index/corpus, just pass
    # the tolerance through, so the only change is the banded-vote aggregation.
    for tol in (1, 2):
        on_recall = {
            name: _mutation_recall(
                off_index, Fingerprinter(), workdir, off_fps, render, ".txt", offset_tolerance=tol
            )
            for name, render in off_mutations
        }
        _record(
            "offset_tolerance",
            str(tol),
            on_recall,
            _false_accept_rate(off_index, off_fps, 0.05, offset_tolerance=tol),
            _false_accept_rate(off_index, off_fps, 0.20, offset_tolerance=tol),
        )

    # candidate_limit: also a SEARCH-time flag (a prefilter). A generous limit
    # must not change recall; a tight one can drop low-overlap matches.
    for limit in (size, 5):
        on_recall = {
            name: _mutation_recall(
                off_index, Fingerprinter(), workdir, off_fps, render, ".txt", candidate_limit=limit
            )
            for name, render in off_mutations
        }
        _record(
            "candidate_limit",
            str(limit),
            on_recall,
            _false_accept_rate(off_index, off_fps, 0.05, candidate_limit=limit),
            _false_accept_rate(off_index, off_fps, 0.20, candidate_limit=limit),
        )

    return comparisons


def confusability_precision(
    rng: np.random.Generator, families: int, siblings_per_family: int, workdir: Path
) -> list[dict[str, object]]:
    """Measure how each collision-adding flag trades precision on sibling docs.

    Builds the confusability corpus (sibling families sharing a header) and, for
    the default config and each ``freq_quantization`` / ``offset_tolerance``
    setting, reports the mean and max impostor (best wrong-file) confidence and
    the leave-one-out false-accept rate at 0.05. A flag that adds collisions
    pushes impostors further over the cutoff -- the precision COST a recall win
    must be weighed against.
    """

    rows: list[dict[str, object]] = []

    def _impostor_stats(
        index: InMemoryHashIndex,
        fingerprints: list[Fingerprint],
        *,
        offset_tolerance: int | None = None,
    ) -> dict[str, object]:
        impostor_conf: list[float] = []
        for fingerprint in fingerprints:
            results = index.search(fingerprint, top_k=5, offset_tolerance=offset_tolerance)
            impostor_conf.append(
                max(
                    (r.confidence for r in results if r.file_id != fingerprint.file_id),
                    default=0.0,
                )
            )
        fa = _false_accept_rate(index, fingerprints, 0.05, offset_tolerance=offset_tolerance)
        fa_strict = _false_accept_rate(index, fingerprints, 0.20, offset_tolerance=offset_tolerance)
        return {
            "mean_impostor": round(statistics.mean(impostor_conf) if impostor_conf else 0.0, 4),
            "max_impostor": round(max(impostor_conf) if impostor_conf else 0.0, 4),
            "false_accept@0.05": round(fa, 4),
            "false_accept@0.20": round(fa_strict, 4),
        }

    # Default config / exact search baseline.
    paths, _texts = _write_sibling_corpus(rng, families, siblings_per_family, workdir)
    base_fp = Fingerprinter()
    base_fps = [base_fp.fingerprint_file(p) for p in paths]
    base_index = InMemoryHashIndex()
    base_index.add_many(base_fps)
    rows.append({"flag": "default", "setting": "off", **_impostor_stats(base_index, base_fps)})

    # offset_tolerance reuses the same index (search-time only).
    for tol in (1, 2):
        rows.append(
            {
                "flag": "offset_tolerance",
                "setting": str(tol),
                **_impostor_stats(base_index, base_fps, offset_tolerance=tol),
            }
        )

    # freq_quantization changes hashes -> rebuild the sibling corpus per q.
    for q in (2, 4):
        fp = Fingerprinter(config=FingerprintConfig(freq_quantization=q))
        q_paths, _ = _write_sibling_corpus(
            np.random.default_rng(_SEED_FOR(rng)), families, siblings_per_family, workdir
        )
        q_fps = [fp.fingerprint_file(p) for p in q_paths]
        q_index = InMemoryHashIndex()
        q_index.add_many(q_fps)
        rows.append({"flag": "freq_quantization", "setting": str(q), **_impostor_stats(q_index, q_fps)})

    return rows


def sweep_image_flags(
    rng: np.random.Generator, size: int, workdir: Path
) -> list[FlagComparison]:
    """OFF (raster) vs ON (phash) sweep on the hard image mutation matrix.

    The hard image mutations are the documented raster weak spots -- strong
    downscale, a 15% border crop, an 8 degree rotation, and a crop+downscale+
    low-quality JPEG -- where the row-major raster signal collapses and a DCT
    pHash, a low-frequency global descriptor, is far more robust.
    """

    import io

    from PIL import Image  # noqa: F401 - gates the case on Pillow

    def _build(mode: Literal["raster", "phash"]) -> tuple[InMemoryHashIndex, list[Fingerprint], list]:
        fp = Fingerprinter(config=FingerprintConfig(image_mode=mode))
        paths, images = _write_hard_image_corpus(np.random.default_rng(_SEED_FOR(rng)), size, workdir)
        fps = [fp.fingerprint_file(p) for p in paths]
        index = InMemoryHashIndex()
        index.add_many(fps)
        return index, fps, images

    def _mutations(images: list) -> list[tuple[str, Callable[[int], bytes], str]]:
        def _resize(i: int) -> bytes:
            buffer = io.BytesIO()
            images[i].resize((64, 48)).save(buffer, format="PNG")
            return buffer.getvalue()

        def _crop(i: int) -> bytes:
            image = images[i]
            width, height = image.size
            box = (int(width * 0.15), int(height * 0.15), int(width * 0.85), int(height * 0.85))
            buffer = io.BytesIO()
            image.crop(box).save(buffer, format="PNG")
            return buffer.getvalue()

        def _rotate(i: int) -> bytes:
            buffer = io.BytesIO()
            images[i].rotate(8, expand=True, fillcolor=128).save(buffer, format="PNG")
            return buffer.getvalue()

        def _jpeg_crop(i: int) -> bytes:
            image = images[i]
            width, height = image.size
            box = (int(width * 0.1), int(height * 0.1), int(width * 0.9), int(height * 0.9))
            buffer = io.BytesIO()
            image.crop(box).resize((90, 70)).save(buffer, format="JPEG", quality=25)
            return buffer.getvalue()

        return [
            ("resize_64x48", _resize, ".png"),
            ("crop_border_15pct", _crop, ".png"),
            ("rotate_8deg", _rotate, ".png"),
            ("jpeg_crop_resize_q25", _jpeg_crop, ".jpg"),
        ]

    off_index, off_fps, off_images = _build("raster")
    off_fp = Fingerprinter(config=FingerprintConfig(image_mode="raster"))
    off_recall = {
        name: _mutation_recall(off_index, off_fp, workdir, off_fps, render, suffix)
        for name, render, suffix in _mutations(off_images)
    }
    off_fa = _false_accept_rate(off_index, off_fps, 0.05)
    off_fa_strict = _false_accept_rate(off_index, off_fps, 0.20)

    on_index, on_fps, on_images = _build("phash")
    on_fp = Fingerprinter(config=FingerprintConfig(image_mode="phash"))
    on_recall = {
        name: _mutation_recall(on_index, on_fp, workdir, on_fps, render, suffix)
        for name, render, suffix in _mutations(on_images)
    }
    on_fa = _false_accept_rate(on_index, on_fps, 0.05)
    on_fa_strict = _false_accept_rate(on_index, on_fps, 0.20)

    return [
        FlagComparison(
            flag="image_mode",
            setting="phash",
            mutation=name,
            off_recall=off_recall[name],
            on_recall=on_recall[name],
            off_false_accept=off_fa,
            on_false_accept=on_fa,
            off_false_accept_strict=off_fa_strict,
            on_false_accept_strict=on_fa_strict,
        )
        for name in off_recall
    ]


def sweep_audio_flags(
    rng: np.random.Generator, size: int, workdir: Path
) -> list[FlagComparison]:
    """OFF (single window) vs ON (window bank) sweep on hard audio excerpts.

    The hard audio mutations are clips/excerpts that re-normalise the whole
    signal and shift the fixed-window time grid (default recall ~0). A bank of
    windows lets the query align at whatever window survives the excerpt.
    """

    import io

    from scipy.io import wavfile

    bank = (512, 1024, 2048, 4096)

    def _build(config: FingerprintConfig) -> tuple[InMemoryHashIndex, list[Fingerprint], list]:
        fp = Fingerprinter(config=config)
        paths, waves = _write_audio_corpus(np.random.default_rng(_SEED_FOR(rng)), size, workdir)
        fps = [fp.fingerprint_file(p) for p in paths]
        index = InMemoryHashIndex()
        index.add_many(fps)
        return index, fps, waves

    def _clip(waves: list, lo: float, hi: float) -> Callable[[int], bytes]:
        def render(i: int) -> bytes:
            sample_rate, data = waves[i]
            start, end = int(len(data) * lo), int(len(data) * hi)
            buffer = io.BytesIO()
            wavfile.write(buffer, sample_rate, data[start:end])
            return buffer.getvalue()

        return render

    off_index, off_fps, off_waves = _build(FingerprintConfig())
    off_fp = Fingerprinter()
    mutations = [
        ("clip_prefix_60pct", _clip(off_waves, 0.0, 0.6)),
        ("excerpt_mid", _clip(off_waves, 0.2, 0.8)),
    ]
    off_recall = {
        name: _mutation_recall(off_index, off_fp, workdir, off_fps, render, ".wav")
        for name, render in mutations
    }
    off_fa = _false_accept_rate(off_index, off_fps, 0.05)
    off_fa_strict = _false_accept_rate(off_index, off_fps, 0.20)

    on_index, on_fps, on_waves = _build(FingerprintConfig(window_bank=bank))
    on_fp = Fingerprinter(config=FingerprintConfig(window_bank=bank))
    on_mutations = [
        ("clip_prefix_60pct", _clip(on_waves, 0.0, 0.6)),
        ("excerpt_mid", _clip(on_waves, 0.2, 0.8)),
    ]
    on_recall = {
        name: _mutation_recall(on_index, on_fp, workdir, on_fps, render, ".wav")
        for name, render in on_mutations
    }
    on_fa = _false_accept_rate(on_index, on_fps, 0.05)
    on_fa_strict = _false_accept_rate(on_index, on_fps, 0.20)

    return [
        FlagComparison(
            flag="window_bank",
            setting=str(bank),
            mutation=name,
            off_recall=off_recall[name],
            on_recall=on_recall[name],
            off_false_accept=off_fa,
            on_false_accept=on_fa,
            off_false_accept_strict=off_fa_strict,
            on_false_accept_strict=on_fa_strict,
        )
        for name in off_recall
    ]


def _SEED_FOR(rng: np.random.Generator) -> int:  # noqa: N802 - reads as a constant helper
    """Draw a fresh deterministic 32-bit seed from the master RNG.

    Each ON config that changes the HASHES (freq_quantization, window_bank,
    image mode) must regenerate its corpus from a generator whose state matches
    the OFF run's, so OFF and ON see byte-identical source files. Drawing the
    sub-seed from the single master ``rng`` keeps the whole sweep a pure function
    of the top-level seed while giving every rebuild the SAME starting state
    (the draw order is fixed), so OFF/ON differ only by the flag.
    """

    return int(rng.integers(0, 2**32))


def run_hard_accuracy(
    *,
    seed: int = 1234,
    text_corpus: int = 30,
    image_corpus: int = 16,
    audio_corpus: int = 12,
    sibling_families: int = 8,
    siblings_per_family: int = 4,
    include_image: bool = True,
    include_audio: bool = True,
) -> dict[str, object]:
    """Run the HARD per-flag sweep and return a structured report dict.

    Deterministic for a fixed ``seed`` (a single master ``default_rng`` drives the
    OFF corpus and all sub-seeds for the hash-changing ON configs). Reports, per
    flag and per hard mutation, recall@1 OFF vs ON and the 0.05 false-accept rate,
    plus a confusability precision table. Image/audio sections skip (and record)
    when their optional dependency is absent.
    """

    report: dict[str, object] = {"seed": seed, "flags": {}, "confusability": [], "skipped": []}
    flags: dict[str, object] = report["flags"]  # type: ignore[assignment]
    skipped: list[str] = report["skipped"]  # type: ignore[assignment]

    with tempfile.TemporaryDirectory(prefix="fp_hard_accuracy_") as tmp:
        workdir = Path(tmp)

        # A single master RNG, drawn from in a fixed order, drives the whole run.
        rng = np.random.default_rng(seed)
        flags["text"] = [c.to_dict() for c in sweep_text_flags(rng, text_corpus, workdir)]
        report["confusability"] = confusability_precision(
            rng, sibling_families, siblings_per_family, workdir
        )

        if include_image:
            try:
                import PIL  # noqa: F401
            except ImportError:
                skipped.append("image (Pillow not installed)")
            else:
                flags["image"] = [c.to_dict() for c in sweep_image_flags(rng, image_corpus, workdir)]

        if include_audio:
            try:
                import scipy  # noqa: F401
            except ImportError:
                skipped.append("audio (scipy not installed)")
            else:
                flags["audio"] = [c.to_dict() for c in sweep_audio_flags(rng, audio_corpus, workdir)]

    return report


def _render_hard_markdown(report: dict[str, object]) -> str:
    """Render the hard per-flag report as a human-readable markdown document."""

    lines: list[str] = [
        f"# Fingerprint engine HARD accuracy report (seed={report['seed']})",
        "",
        "Per-flag recall@1 OFF vs ON on a deliberately hard near-dup corpus, with "
        "the leave-one-out false-accept rate at the default 0.05 cutoff. A flag is "
        "RECALL-justified only where on_recall > off_recall at acceptable precision.",
        "",
    ]
    flags: dict[str, list[dict]] = report["flags"]  # type: ignore[assignment]
    for handler, comparisons in flags.items():
        lines.append(f"## {handler}")
        lines.append("")
        lines.append(
            "| flag | setting | mutation | off recall | on recall | delta | "
            "on FA@0.05 | on FA@0.20 |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for c in comparisons:
            mark = " **" if c["recall_delta"] > 0 else " "
            lines.append(
                f"| {c['flag']} | {c['setting']} | {c['mutation']} | {c['off_recall']} |"
                f"{mark}{c['on_recall']}{'**' if c['recall_delta'] > 0 else ''} | "
                f"{c['recall_delta']:+} | {c['on_false_accept@0.05']} | {c['on_false_accept@0.20']} |"
            )
        lines.append("")

    rows: list[dict] = report["confusability"]  # type: ignore[assignment]
    if rows:
        lines.append("## confusability (sibling docs): impostor confidence + precision cost")
        lines.append("")
        lines.append("| flag | setting | mean impostor | max impostor | FA@0.05 | FA@0.20 |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for row in rows:
            lines.append(
                f"| {row['flag']} | {row['setting']} | {row['mean_impostor']} | "
                f"{row['max_impostor']} | {row['false_accept@0.05']} | {row['false_accept@0.20']} |"
            )
        lines.append("")

    skipped: list[str] = report["skipped"]  # type: ignore[assignment]
    if skipped:
        lines.append("## skipped")
        lines.append("")
        for item in skipped:
            lines.append(f"- {item}")
        lines.append("")
    return "\n".join(lines)


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

    skipped: list[str] = report["skipped"]  # type: ignore[assignment]
    if skipped:
        lines.append("## skipped")
        lines.append("")
        for item in skipped:
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
    parser.add_argument(
        "--mode",
        choices=("standard", "hard"),
        default="standard",
        help="standard near-dup matrix (default) or the HARD per-flag recall/precision sweep",
    )
    args = parser.parse_args(argv)

    if args.mode == "hard":
        report = run_hard_accuracy(
            seed=args.seed,
            include_image=not args.no_image,
            include_audio=not args.no_audio,
        )
        renderer = _render_hard_markdown
    else:
        report = run_accuracy(
            seed=args.seed,
            text_corpus=args.text_corpus,
            image_corpus=args.image_corpus,
            audio_corpus=args.audio_corpus,
            include_image=not args.no_image,
            include_audio=not args.no_audio,
        )
        renderer = _render_markdown

    if args.format == "markdown":
        print(renderer(report))
    else:
        print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
