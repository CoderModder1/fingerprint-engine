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

Usage::

    python benchmarks/heavy_handlers.py            # both sections, JSON to stdout
    python benchmarks/heavy_handlers.py --no-video # embedding section only

The video section is skipped (with a note in the JSON) when ``av`` / ``imageio``
are not importable, so a core-only environment still runs the embedding section.
"""

from __future__ import annotations

import argparse
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

from fingerprint_engine.core.fingerprinter import Fingerprinter
from fingerprint_engine.core.index import InMemoryHashIndex
from fingerprint_engine.core.models import Fingerprint, FingerprintConfig

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
            # Append MANY vectors -- crosses an adaptive-window boundary (see verdict).
            ("append_30", np.vstack([base, _make_embedding_doc(99, 30, dim)])),
            # Delete a few rows.
            ("delete_5_rows", np.delete(base, [10, 11, 12, 30, 31], axis=0)),
            # Insert a few rows mid-stream.
            ("insert_3_rows", np.insert(base, 20, _make_embedding_doc(77, 3, dim), axis=0)),
            # Perturb (small gaussian) a few rows.
            ("perturb_5_rows", _perturb_rows(base, [5, 6, 7, 8, 9], scale=0.1)),
            # Reorder a contiguous block (swap two adjacent runs of rows).
            ("reorder_block", np.vstack([base[:15], base[25:35], base[15:25], base[35:]])),
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
                    "effective_window": int(fp.metadata.get("effective_window_size", 0)),
                }
            )

        unrelated = _make_embedding_doc(seed=4242, num_vectors=num_vectors, dim=dim)
        unrelated_path = tmp_path / "unrelated.npy"
        np.save(unrelated_path, unrelated)
        unrelated_fp = fingerprinter.fingerprint_file(unrelated_path)
        _m, _c, impostor_top = _query(index, unrelated_fp, base_id)

    summary = _summarize_variants(variant_rows)
    return {
        "corpus": {"docs": num_docs, "vectors_per_doc": num_vectors, "dim": dim},
        "self_recall_at_1": round(self_hits / num_docs, 4),
        "mean_self_confidence": round(statistics.mean(self_confs), 4),
        "near_dup_recall_at_1": summary["near_dup_recall_at_1"],
        "mean_near_dup_confidence": summary["mean_confidence"],
        "per_variant": summary["per_variant"],
        "max_impostor_confidence": round(impostor_top, 4),
        "throughput_docs_per_s": round(num_docs / fp_elapsed, 3) if fp_elapsed > 0 else None,
        "avg_hashes_per_file": round(statistics.mean(hash_counts), 1),
        "original_effective_windows": sorted(set(eff_windows)),
        "window_note": (
            "EmbeddingFileHandler defines no class default_signal_window, so the "
            "signal is fingerprinted at the GLOBAL default window (4096), which is "
            "LENGTH-ADAPTIVE -- not the per-frame d. The effective window therefore "
            "depends on the signal length (num_vectors*d), so a length-changing "
            "near-dup that crosses an adaptive-window boundary lands on a different "
            "time grid and fails to align (see the append_30 variant)."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-video", action="store_true", help="skip the video section")
    parser.add_argument("--no-embedding", action="store_true", help="skip the embedding section")
    args = parser.parse_args()

    report: dict[str, object] = {}
    if not args.no_video:
        report["video"] = run_video_benchmark()
    if not args.no_embedding:
        report["embedding"] = run_embedding_benchmark()
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
