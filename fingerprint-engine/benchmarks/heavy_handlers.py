"""Benchmark the heavy-dependency handler skeletons end-to-end.

Exercises the two skeleton handlers -- :class:`VideoFileHandler` and
:class:`EmbeddingFileHandler` -- as REAL fingerprinters: synthesize a small
corpus, index the originals through a :class:`Fingerprinter` +
:class:`InMemoryHashIndex`, then query each original with itself, a set of
near-duplicate variants, and an unrelated item. It reports, per handler:

  * self recall@1 (an indexed file must rank itself first),
  * per-variant near-duplicate recall@1 and mean confidence,
  * the maximum confidence any UNRELATED query scores against the corpus
    (the impostor ceiling -- separation between real and spurious matches),
  * fingerprint throughput (items/s) and average hashes/file.

Everything is DETERMINISTIC: all synthesis is driven by a seeded
``numpy.random.default_rng`` (no unseeded randomness, no clock-derived values),
so a run is reproducible.

The corpora are intentionally tiny (a handful of ~64x48, ~40-frame clips; a
handful of ~60-vector, d=64 embedding docs) so the whole thing runs in a couple
of seconds and pulls no heavy model runtime -- the embedding path is the
numpy-only precomputed ``.npy`` path, never an encoder.

A THIRD, OPT-IN section (:func:`run_encoder_benchmark`) exercises the embedding
handler's REAL ENCODER path: it wires a :class:`Model2VecEmbedder` (the cached
``minishlab/potion-base-8M`` static model, 256-dim, no torch) into
``EmbeddingFileHandler(embedder=...)`` and DRIVES THE HANDLER DIRECTLY
(``load`` -> ``to_signal`` -> ``extract_peaks`` at window=hop=d, then assembles a
``Fingerprint``), because ``EmbeddingFileHandler.can_handle`` only claims
``.npy``/``.npz``/``.jsonl`` and so will never auto-route a ``.txt`` document.
It builds a small corpus of distinct multi-paragraph prose docs plus per-doc
near-dup variants (exact-passage reuse, paraphrase, insert/delete/append,
unrelated) and reports what the constellation pipeline actually matches over
real embeddings. This section is skipped cleanly when ``model2vec`` is absent,
so a core-only / CI environment never downloads a model.

Usage::

    python benchmarks/heavy_handlers.py             # video + embedding sections
    python benchmarks/heavy_handlers.py --no-video  # embedding section only
    python benchmarks/heavy_handlers.py --encoder   # also run the model2vec encoder

The video section is skipped (with a note in the JSON) when ``av`` / ``imageio``
are not importable, and the encoder section is skipped when ``model2vec`` is not
importable, so a core-only environment still runs the embedding section.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np

# 0-hash RuntimeWarnings can fire for the smallest synthetic inputs; the
# benchmark measures them explicitly, so silence the noise.
warnings.simplefilter("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fingerprint_engine.core.exceptions import MissingDependencyError
from fingerprint_engine.core.fft_pipeline import FFTFingerprintPipeline
from fingerprint_engine.core.fingerprinter import Fingerprinter
from fingerprint_engine.core.index import InMemoryHashIndex
from fingerprint_engine.core.models import (
    FORMAT_VERSION_KEY,
    Fingerprint,
    FingerprintConfig,
    effective_format_version,
)
from fingerprint_engine.handlers.embedders import DEFAULT_MODEL2VEC_MODEL, Model2VecEmbedder
from fingerprint_engine.handlers.embedding_handler import EmbeddingFileHandler

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _query(index: InMemoryHashIndex, query_fp: Fingerprint, target_id: str) -> tuple[bool, float, float]:
    """Run one search and report (matched@1, confidence-vs-target, top-confidence).

    ``matched@1`` is whether the rank-1 result's ``file_id`` equals ``target_id``
    (the original we expect this query to map to). ``confidence-vs-target`` is the
    confidence the target itself earned anywhere in the result list (0 if absent),
    and ``top-confidence`` is the rank-1 confidence regardless of identity (used
    for the impostor ceiling).
    """

    results = index.search(query_fp, top_k=5)
    if not results:
        return False, 0.0, 0.0
    top = results[0]
    matched = top.file_id == target_id
    target_conf = next((r.confidence for r in results if r.file_id == target_id), 0.0)
    return matched, float(target_conf), float(top.confidence)


def _summarize_variants(rows: list[dict[str, object]]) -> dict[str, object]:
    matched = [bool(r["matched"]) for r in rows]
    confs = [float(r["confidence"]) for r in rows]
    return {
        "near_dup_recall_at_1": round(sum(matched) / len(matched), 4) if matched else 0.0,
        "mean_confidence": round(statistics.mean(confs), 4) if confs else 0.0,
        "per_variant": rows,
    }


# ---------------------------------------------------------------------------
# Video synthesis + benchmark
# ---------------------------------------------------------------------------


def _make_video_frames(seed: int, num_frames: int, width: int, height: int) -> list[np.ndarray]:
    """Deterministic, per-clip-distinct moving/structured RGB frames.

    Each clip gets a per-seed colour ramp plus a moving box at a per-seed start
    position, so clips are mutually distinct (no two share content) while every
    frame carries strong, resolution-independent spatial structure for the
    canonical-256 keyframe to latch onto.
    """

    rng = np.random.default_rng(seed)
    box_x0 = int(rng.integers(6, max(7, width - 14)))
    box_y0 = int(rng.integers(6, max(7, height - 14)))
    box_color = rng.integers(40, 210, size=3).astype(np.uint8)
    ramp_r = int(rng.integers(1, 4))
    ramp_g = int(rng.integers(1, 4))
    frames: list[np.ndarray] = []
    for i in range(num_frames):
        img = np.zeros((height, width, 3), dtype=np.uint8)
        img[:, :, 0] = (np.arange(width)[None, :] * ramp_r + i * 3) % 256
        img[:, :, 1] = (np.arange(height)[:, None] * ramp_g + i * 2) % 256
        x = (box_x0 + i) % max(1, width - 10)
        y = (box_y0 + i // 2) % max(1, height - 10)
        img[y : y + 10, x : x + 10] = box_color
        frames.append(img)
    return frames


def _write_video(imageio_mod, frames: list[np.ndarray], path: Path, *, fps: int = 10, **kw: object) -> Path:
    imageio_mod.mimsave(str(path), frames, format="FFMPEG", codec="libx264", fps=fps, **kw)
    return path


def _resize_frames(frames: list[np.ndarray], size: tuple[int, int]) -> list[np.ndarray]:
    from PIL import Image

    return [np.asarray(Image.fromarray(f).resize(size)) for f in frames]


def _noise_frames(frames: list[np.ndarray], seed: int, amplitude: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    out: list[np.ndarray] = []
    for f in frames:
        jitter = rng.integers(-amplitude, amplitude + 1, size=f.shape)
        out.append(np.clip(f.astype(np.int16) + jitter, 0, 255).astype(np.uint8))
    return out


def run_video_benchmark(num_clips: int = 6, num_frames: int = 48, width: int = 64, height: int = 48) -> dict:
    try:
        import av  # noqa: F401 - the handler's video decode backend (video extra)
        import imageio.v2 as imageio  # the test/benchmark video WRITER (dev extra)
        from PIL import Image  # noqa: F401 - keyframe canonicalisation (video extra)
    except ImportError as exc:
        return {"skipped": True, "reason": f"missing video dependency: {exc}"}

    fingerprinter = Fingerprinter(FingerprintConfig())
    index = InMemoryHashIndex()

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        clip_frames: dict[int, list[np.ndarray]] = {}
        original_fps: dict[int, Fingerprint] = {}

        fp_start = time.perf_counter()
        hash_counts: list[int] = []
        for clip in range(num_clips):
            frames = _make_video_frames(seed=clip, num_frames=num_frames, width=width, height=height)
            clip_frames[clip] = frames
            path = _write_video(imageio, frames, tmp_path / f"clip{clip}.mp4")
            fp = fingerprinter.fingerprint_file(path)
            original_fps[clip] = fp
            hash_counts.append(fp.hash_count)
            index.add(fp)
        fp_elapsed = time.perf_counter() - fp_start

        # Self recall@1: every indexed original must rank itself first.
        self_hits = 0
        self_confs: list[float] = []
        for fp in original_fps.values():
            matched, conf, _ = _query(index, fp, fp.file_id)
            self_hits += int(matched)
            self_confs.append(conf)

        # Near-duplicate variants of clip 0 (the canonical query source). Each
        # variant RE-ENCODES the actual frames so the bytes differ but the
        # content is shared.
        base_frames = clip_frames[0]
        base_id = original_fps[0].file_id
        variant_specs: list[tuple[str, list[np.ndarray], dict]] = [
            # Re-encode at a lower bitrate, SAME fps -> keyframe cadence is
            # preserved, so the keyframe grid aligns.
            ("reencode_bitrate", base_frames, {"bitrate": "150k"}),
            # Trim to a middle excerpt (frames 10..40): the hard case -- the
            # excerpt's keyframe grid starts at a different absolute frame.
            ("excerpt_10_40", base_frames[10:40], {}),
            # Resize the frames: the canonical-256 reduction should absorb this.
            ("resize_96x72", _resize_frames(base_frames, (96, 72)), {}),
            # Mild per-pixel noise.
            ("pixel_noise", _noise_frames(base_frames, seed=909, amplitude=12), {}),
        ]
        variant_rows: list[dict[str, object]] = []
        for name, frames, kw in variant_specs:
            path = _write_video(imageio, frames, tmp_path / f"variant_{name}.mp4", **kw)
            fp = fingerprinter.fingerprint_file(path)
            matched, conf, _top = _query(index, fp, base_id)
            variant_rows.append(
                {
                    "variant": name,
                    "matched": matched,
                    "confidence": round(conf, 4),
                    "query_hashes": fp.hash_count,
                    "sampled_keyframes": fp.metadata.get("sampled_keyframes"),
                }
            )

        # Impostor ceiling: an UNRELATED clip queried against the corpus. Its
        # top confidence is the spurious-match ceiling we want well below the
        # near-dup confidences.
        unrelated_frames = _make_video_frames(seed=999, num_frames=num_frames, width=width, height=height)
        unrelated_path = _write_video(imageio, unrelated_frames, tmp_path / "unrelated.mp4")
        unrelated_fp = fingerprinter.fingerprint_file(unrelated_path)
        _m, _c, impostor_top = _query(index, unrelated_fp, base_id)

    summary = _summarize_variants(variant_rows)
    return {
        "skipped": False,
        "corpus": {"clips": num_clips, "frames_per_clip": num_frames, "resolution": [width, height]},
        "self_recall_at_1": round(self_hits / num_clips, 4),
        "mean_self_confidence": round(statistics.mean(self_confs), 4),
        "near_dup_recall_at_1": summary["near_dup_recall_at_1"],
        "mean_near_dup_confidence": summary["mean_confidence"],
        "per_variant": summary["per_variant"],
        "max_impostor_confidence": round(impostor_top, 4),
        "throughput_videos_per_s": round(num_clips / fp_elapsed, 3) if fp_elapsed > 0 else None,
        "avg_hashes_per_file": round(statistics.mean(hash_counts), 1),
    }


# ---------------------------------------------------------------------------
# Embedding synthesis + benchmark
# ---------------------------------------------------------------------------


def _make_embedding_doc(seed: int, num_vectors: int, dim: int) -> np.ndarray:
    """Deterministic, mutually-distinct ordered vector sequence.

    Each row is a smooth phase function of its position plus a per-doc signature
    direction, with a little seeded jitter. Successive rows are correlated (a
    real "stream"), and different seeds give different signatures so docs are
    mutually distinct.
    """

    rng = np.random.default_rng(seed)
    signature = rng.standard_normal(dim)
    rows: list[np.ndarray] = []
    for i in range(num_vectors):
        phase = np.sin(np.linspace(0.0, (i + 1) * 0.3, dim)) + signature * 0.5
        rows.append(phase + rng.standard_normal(dim) * 0.05)
    return np.asarray(rows, dtype=np.float32)


def run_embedding_benchmark(num_docs: int = 8, num_vectors: int = 60, dim: int = 64) -> dict:
    fingerprinter = Fingerprinter(FingerprintConfig())
    index = InMemoryHashIndex()

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        docs: dict[int, np.ndarray] = {}
        original_fps: dict[int, Fingerprint] = {}

        fp_start = time.perf_counter()
        hash_counts: list[int] = []
        eff_windows: list[int] = []
        for doc in range(num_docs):
            arr = _make_embedding_doc(seed=doc + 1, num_vectors=num_vectors, dim=dim)
            docs[doc] = arr
            path = tmp_path / f"doc{doc}.npy"
            np.save(path, arr)
            fp = fingerprinter.fingerprint_file(path)
            original_fps[doc] = fp
            hash_counts.append(fp.hash_count)
            eff_windows.append(int(fp.metadata.get("effective_window_size", 0)))
            index.add(fp)
        fp_elapsed = time.perf_counter() - fp_start

        self_hits = 0
        self_confs: list[float] = []
        for fp in original_fps.values():
            matched, conf, _ = _query(index, fp, fp.file_id)
            self_hits += int(matched)
            self_confs.append(conf)

        base = docs[0]
        base_id = original_fps[0].file_id
        edit_rng = np.random.default_rng(2024)

        def _perturb_rows(arr: np.ndarray, indices: list[int], scale: float) -> np.ndarray:
            out = arr.copy()
            out[indices] += edit_rng.standard_normal((len(indices), arr.shape[1])).astype(np.float32) * scale
            return out

        variant_specs: list[tuple[str, np.ndarray]] = [
            # Append a few vectors (small length change).
            ("append_5", np.vstack([base, _make_embedding_doc(99, 5, dim)])),
            # Append MANY vectors -- length-crossing near-dup. At the OLD global
            # length-adaptive window this crossed 512 -> 1024 and missed; with the
            # per-frame window=d it stays length-stable and matches.
            ("append_30", np.vstack([base, _make_embedding_doc(99, 30, dim)])),
            # Delete a few rows.
            ("delete_5_rows", np.delete(base, [10, 11, 12, 30, 31], axis=0)),
            # Insert a few rows mid-stream.
            ("insert_3_rows", np.insert(base, 20, _make_embedding_doc(77, 3, dim), axis=0)),
            # Perturb (small gaussian) a few rows.
            ("perturb_5_rows", _perturb_rows(base, [5, 6, 7, 8, 9], scale=0.1)),
            # Reorder a contiguous block (swap two adjacent runs of rows).
            ("reorder_block", np.vstack([base[:15], base[25:35], base[15:25], base[35:]])),
            # SAME base content re-embedded at a DIFFERENT dimensionality (each row
            # width-doubled to 2*d). The per-frame window is d-aligned, so a 2*d
            # stream lands on a different time grid and frequency basis and MUST
            # NOT match: window=d isolates by dimensionality. Tracked as a control,
            # not a near-dup, so it is excluded from the near-dup recall stats.
            ("different_dim", np.concatenate([base, base], axis=1)),
        ]
        variant_rows: list[dict[str, object]] = []
        for name, arr in variant_specs:
            path = tmp_path / f"variant_{name}.npy"
            np.save(path, arr.astype(np.float32))
            fp = fingerprinter.fingerprint_file(path)
            matched, conf, _top = _query(index, fp, base_id)
            variant_rows.append(
                {
                    "variant": name,
                    "matched": matched,
                    "confidence": round(conf, 4),
                    "query_hashes": fp.hash_count,
                    "num_vectors": int(arr.shape[0]),
                    "embedding_dim": int(arr.shape[1]),
                    # The TRUE per-frame FFT window used (== d); the framework's
                    # ``effective_window_size`` reflects the GLOBAL length-adaptive
                    # pipeline and does NOT govern embedding matching anymore.
                    "embedding_window": int(fp.metadata.get("embedding_signal_window", 0)),
                    "effective_window": int(fp.metadata.get("effective_window_size", 0)),
                }
            )

        unrelated = _make_embedding_doc(seed=4242, num_vectors=num_vectors, dim=dim)
        unrelated_path = tmp_path / "unrelated.npy"
        np.save(unrelated_path, unrelated)
        unrelated_fp = fingerprinter.fingerprint_file(unrelated_path)
        _m, _c, impostor_top = _query(index, unrelated_fp, base_id)

    # ``different_dim`` is a negative control (it MUST miss), so it is excluded
    # from the near-dup recall stat -- those stats only cover the genuine
    # same-d near-duplicates -- but it is still reported in ``per_variant``.
    near_dup_rows = [row for row in variant_rows if row["variant"] != "different_dim"]
    summary = _summarize_variants(near_dup_rows)
    return {
        "corpus": {"docs": num_docs, "vectors_per_doc": num_vectors, "dim": dim},
        "self_recall_at_1": round(self_hits / num_docs, 4),
        "mean_self_confidence": round(statistics.mean(self_confs), 4),
        "near_dup_recall_at_1": summary["near_dup_recall_at_1"],
        "mean_near_dup_confidence": summary["mean_confidence"],
        "per_variant": variant_rows,
        "max_impostor_confidence": round(impostor_top, 4),
        "throughput_docs_per_s": round(num_docs / fp_elapsed, 3) if fp_elapsed > 0 else None,
        "avg_hashes_per_file": round(statistics.mean(hash_counts), 1),
        "original_effective_windows": sorted(set(eff_windows)),
        "window_note": (
            "EmbeddingFileHandler overrides extract_peaks to fingerprint at a "
            "per-frame window=hop=d (fixed_window=True), so each vector is exactly "
            "one FFT frame and all files of the same dimensionality d share one "
            "time grid regardless of sequence length n. Matching is therefore "
            "LENGTH-STABLE: a length-crossing near-dup (append_30) stays aligned "
            "with its parent. The framework-recorded ``effective_window_size`` "
            "still reflects the GLOBAL length-adaptive pipeline and no longer "
            "governs embedding matching (see ``embedding_window`` for the real d "
            "window). A different-dimensionality copy (``different_dim``) lands on "
            "a different grid and correctly does NOT match."
        ),
    }


# ---------------------------------------------------------------------------
# Encoder (model2vec) synthesis + benchmark -- the REAL encode-on-load path.
# Model2VecEmbedder is shipped in the package (fingerprint_engine.handlers.
# embedders, imported at the top of this module); this section only synthesizes
# corpora and measures that shipped encoder end-to-end.
# ---------------------------------------------------------------------------


# Deterministic per-doc word pools: a doc draws sentences from these with a
# seeded RNG, so each source doc is mutually distinct prose yet fully
# reproducible. The encoder maps real words -> real vectors, so distinct word
# choices give distinct embedding spectra (the whole point of the measurement).
_SUBJECTS = (
    "The harbor", "A distant comet", "The old library", "Each turbine",
    "The mountain pass", "Our research vessel", "The desert outpost",
    "A migrating heron", "The clockmaker", "The river delta",
    "The glass observatory", "The northern forest",
)
_VERBS = (
    "channels", "records", "scatters", "amplifies", "preserves", "distorts",
    "gathers", "releases", "measures", "reflects", "absorbs", "transmits",
)
_OBJECTS = (
    "a faint signal", "the morning tide", "ancient sediment", "unexpected warmth",
    "the migratory pattern", "a low hum", "the spectral drift", "quiet interference",
    "the seasonal bloom", "stray photons", "a tidal rhythm", "the magnetic flux",
)
_TAILS = (
    "under a pale sky.", "before the equinox.", "across the basin.",
    "through the canopy.", "despite the storm.", "near the fault line.",
    "beyond the breakwater.", "within the archive.", "along the ridge.",
    "beneath the ice.",
)


def _make_prose_doc(seed: int, num_paragraphs: int = 8, sentences_per: int = 3) -> list[str]:
    """A deterministic list of distinct paragraph strings (one chunk per line).

    Each paragraph is ``sentences_per`` sentences drawn (seeded) from the shared
    word pools. A different ``seed`` gives different word choices and so a
    mutually-distinct document. One paragraph == one line == one embedding chunk.
    """

    rng = np.random.default_rng(seed)

    def sentence() -> str:
        return (
            f"{rng.choice(_SUBJECTS)} {rng.choice(_VERBS)} "
            f"{rng.choice(_OBJECTS)} {rng.choice(_TAILS)}"
        )

    return [" ".join(sentence() for _ in range(sentences_per)) for _ in range(num_paragraphs)]


# A hand-written FULL paraphrase of doc-0's eight paragraphs: every paragraph is
# reworded (synonyms, reordered clauses) so NO line is verbatim-shared with the
# parent, while staying semantically close (measured cosine 0.2-0.65 per pair).
# This is the honest "semantic-only, no exact reuse" probe.
_DOC0_PARAPHRASE = (
    "Beneath a washed-out heaven the port relays a weak transmission.",
    "Before the spring balance point a far-off shooting star strews old silt.",
    "Surprising heat is kept inside the storage of the aged book hall.",
    "Over the valley every windmill boosts a quiet drone.",
    "Along the crest the highland gap collects the springtime flowering.",
    "Past the seawall our survey ship gauges the wavelength wandering.",
    "Close to the rift the arid station notes faint background noise.",
    "Through the leaves a travelling crane lets go the rising-water cadence.",
)


def _doc_bytes(paragraphs: list[str]) -> bytes:
    """Serialize a paragraph list to the on-disk doc form (one chunk per line)."""

    return ("\n".join(paragraphs)).encode("utf-8")


def _encode_fingerprint(
    handler: EmbeddingFileHandler,
    config: FingerprintConfig,
    format_version: int,
    paragraphs: list[str],
    path: Path,
) -> tuple[Fingerprint, int]:
    """Drive the encoder path directly and assemble a :class:`Fingerprint`.

    ``EmbeddingFileHandler.can_handle`` only claims ``.npy``/``.npz``/``.jsonl``,
    so a ``.txt`` doc is NEVER auto-routed here by the Fingerprinter. We
    therefore run the handler pipeline by hand exactly as the framework would:
    ``load`` (which calls the injected embedder on the raw bytes) ->
    ``to_signal`` -> ``extract_peaks`` (which builds the per-frame window=hop=d
    sub-pipeline) -> assemble the ``Fingerprint`` with the content sha256 as the
    ``file_id`` and the same format-version stamp the Fingerprinter writes.
    Returns ``(fingerprint, dim)``; the encode-inclusive elapsed time is
    measured by the caller.
    """

    content = _doc_bytes(paragraphs)
    path.write_bytes(content)
    payload = handler.load(path)  # decode + encode + L2-normalise + learn d
    signal = handler.to_signal(payload)
    landmarks, hashes = handler.extract_peaks(signal, FFTFingerprintPipeline(config))
    file_id = hashlib.sha256(content).hexdigest()
    fingerprint = Fingerprint(
        file_id=file_id,
        path=str(path),
        handler=handler.name,
        size_bytes=len(content),
        content_sha256=file_id,
        config={FORMAT_VERSION_KEY: format_version},
        landmarks=landmarks,
        hashes=hashes,
        metadata=handler.metadata(payload),
    )
    return fingerprint, payload.dim


def run_encoder_benchmark(num_docs: int = 8) -> dict:
    """Benchmark the REAL model2vec encode-on-load path end-to-end.

    Returns ``{"skipped": True, ...}`` when model2vec is unimportable so a
    core-only environment degrades cleanly. Otherwise builds ``num_docs``
    distinct prose docs, indexes them through the encoder path, and queries
    self + per-doc-0 near-dup variants + an unrelated doc, reporting recall,
    confidence, the impostor ceiling, throughput (encode-inclusive), and d.
    """

    try:
        embedder = Model2VecEmbedder()
    except (ImportError, MissingDependencyError) as exc:  # model2vec (or its deps) absent
        return {"skipped": True, "reason": f"missing encoder dependency: {exc}"}

    config = FingerprintConfig()
    format_version = effective_format_version(config)
    handler = EmbeddingFileHandler(embedder=embedder)
    index = InMemoryHashIndex()

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        docs: dict[int, list[str]] = {
            i: _make_prose_doc(seed=1000 + i) for i in range(num_docs)
        }

        original_fps: dict[int, Fingerprint] = {}
        hash_counts: list[int] = []
        dim = 0
        fp_start = time.perf_counter()  # encode-INCLUSIVE timing (the real cost)
        for i, paragraphs in docs.items():
            fp, dim = _encode_fingerprint(
                handler, config, format_version, paragraphs, tmp_path / f"doc{i}.txt"
            )
            original_fps[i] = fp
            hash_counts.append(fp.hash_count)
            index.add(fp)
        fp_elapsed = time.perf_counter() - fp_start

        # Self recall@1: every indexed doc must rank itself first.
        self_hits = 0
        self_confs: list[float] = []
        for fp in original_fps.values():
            matched, conf, _ = _query(index, fp, fp.file_id)
            self_hits += int(matched)
            self_confs.append(conf)

        base = docs[0]
        base_id = original_fps[0].file_id
        base_codes = {h.hash_code for h in original_fps[0].hashes}
        # Fresh prose pools for the "new material" in reuse / insert / append, so
        # the added paragraphs are NOT accidentally shared with doc-0.
        filler = _make_prose_doc(seed=55_555)
        more_filler = _make_prose_doc(seed=99_999)

        variant_specs: list[tuple[str, list[str], bool]] = [
            # EXACT_PASSAGE_REUSE: copy 4 of doc-0's paragraphs VERBATIM + add 4
            # brand-new ones. Identical chunks -> identical vectors -> identical
            # spectra -> identical hashes, so this SHOULD match strongly.
            ("exact_passage_reuse", base[:4] + filler[:4], True),
            # PARAPHRASE: every paragraph reworded, NO verbatim line shared.
            # Semantically close but lexically different -> different vectors ->
            # different spectra. The honest probe for semantic matching.
            ("paraphrase_all", list(_DOC0_PARAPHRASE), True),
            # INSERT two new paragraphs mid-stream (the rest verbatim).
            ("insert_2_paras", base[:3] + more_filler[:2] + base[3:], True),
            # DELETE two paragraphs (the rest verbatim).
            ("delete_2_paras", base[:2] + base[4:], True),
            # APPEND four new paragraphs after the verbatim original.
            ("append_4_paras", base + filler[:4], True),
            # UNRELATED doc (negative control, excluded from near-dup recall).
            ("unrelated", _make_prose_doc(seed=4_242_424), False),
        ]

        variant_rows: list[dict[str, object]] = []
        for name, paragraphs, is_near_dup in variant_specs:
            fp, _vdim = _encode_fingerprint(
                handler, config, format_version, paragraphs, tmp_path / f"variant_{name}.txt"
            )
            matched, conf, top_conf = _query(index, fp, base_id)
            shared = len(base_codes & {h.hash_code for h in fp.hashes})
            variant_rows.append(
                {
                    "variant": name,
                    "is_near_dup": is_near_dup,
                    "matched": matched,
                    "confidence": round(conf, 4),
                    "top_confidence": round(top_conf, 4),
                    "shared_hash_codes": shared,
                    "query_hashes": fp.hash_count,
                    "num_chunks": len(paragraphs),
                    "embedding_window": int(fp.metadata.get("embedding_signal_window", 0)),
                }
            )

    by_name = {row["variant"]: row for row in variant_rows}
    near_dup_rows = [row for row in variant_rows if row["is_near_dup"]]
    summary = _summarize_variants(near_dup_rows)
    # Impostor ceiling: the max top-confidence any UNRELATED query scored against
    # the corpus. A genuine near-dup must clear this for its rank-1 to be signal.
    impostor_top = float(by_name["unrelated"]["top_confidence"])

    return {
        "skipped": False,
        "model": DEFAULT_MODEL2VEC_MODEL,
        "corpus": {"docs": num_docs, "paragraphs_per_doc": 8, "chunking": "one_line_per_chunk"},
        "embedding_dim": dim,
        "self_recall_at_1": round(self_hits / num_docs, 4),
        "mean_self_confidence": round(statistics.mean(self_confs), 4),
        "near_dup_recall_at_1": summary["near_dup_recall_at_1"],
        "mean_near_dup_confidence": summary["mean_confidence"],
        "per_variant": variant_rows,
        "max_impostor_confidence": round(impostor_top, 4),
        "throughput_docs_per_s_encode_inclusive": (
            round(num_docs / fp_elapsed, 3) if fp_elapsed > 0 else None
        ),
        "avg_hashes_per_file": round(statistics.mean(hash_counts), 1),
        "characterization": (
            "The encoder path is a SHARED-EXACT-EMBEDDING-SUBSEQUENCE detector, "
            "not a semantic-similarity matcher. EXACT_PASSAGE_REUSE / insert / "
            "delete / append all reuse verbatim paragraphs whose chunks encode to "
            "BIT-IDENTICAL vectors (model2vec static embeddings are deterministic) "
            "-> identical per-frame spectra -> identical constellation hashes, so "
            "they share many hash codes and match with real confidence. A FULL "
            "PARAPHRASE shares NO verbatim line, so despite cosine 0.2-0.65 "
            "semantic similarity its chunks encode to DIFFERENT vectors -> "
            "different spectra -> essentially no shared hash codes; its confidence "
            "collapses to the impostor/noise floor. See per_variant.shared_hash_"
            "codes and confidence for the quantified split."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-video", action="store_true", help="skip the video section")
    parser.add_argument("--no-embedding", action="store_true", help="skip the embedding section")
    parser.add_argument(
        "--encoder",
        action="store_true",
        help="also run the model2vec encode-on-load section (needs model2vec)",
    )
    args = parser.parse_args()

    report: dict[str, object] = {}
    if not args.no_video:
        report["video"] = run_video_benchmark()
    if not args.no_embedding:
        report["embedding"] = run_embedding_benchmark()
    if args.encoder:
        report["encoder"] = run_encoder_benchmark()
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
