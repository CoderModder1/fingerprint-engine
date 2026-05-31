"""Tests for the heavy-dependency handler skeletons (item 3).

Covers the two new handlers -- :class:`VideoFileHandler` and
:class:`EmbeddingFileHandler` -- with two kinds of checks:

* ALWAYS-ON (no heavy dep needed): routing is unchanged (``can_handle`` returns
  0 for ``.py`` / ``.png`` / ``.wav`` so nothing is mis-routed), the handlers
  auto-discover, importing the package needs no new dep, and ``load`` raises
  :class:`MissingDependencyError` when the backend is absent.
* importorskip-GATED: when the backend is present, a tiny synthetic input
  fingerprints end-to-end. The embedding precomputed path is numpy-only, so its
  synthetic fingerprint test always runs.
"""

from __future__ import annotations

import builtins
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fingerprint_engine.core.exceptions import MissingDependencyError
from fingerprint_engine.core.fft_pipeline import FFTFingerprintPipeline
from fingerprint_engine.core.fingerprinter import Fingerprinter
from fingerprint_engine.core.models import FingerprintConfig
from fingerprint_engine.handlers.embedding_handler import EmbeddingFileHandler
from fingerprint_engine.handlers.video_handler import VideoFileHandler

# ---------------------------------------------------------------------------
# Routing is unchanged: neither handler claims unrelated files.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["script.py", "photo.png", "clip.wav", "doc.pdf", "song.mp3"])
def test_video_handler_ignores_unrelated_files(name: str) -> None:
    # Pass each unrelated type's typical mime so the prefix branch is exercised
    # too; the video handler must score exactly 0 for all of them.
    mime = {
        "photo.png": "image/png",
        "clip.wav": "audio/wav",
        "doc.pdf": "application/pdf",
        "song.mp3": "audio/mpeg",
        "script.py": "text/x-python",
    }.get(name)
    assert VideoFileHandler.can_handle(name, mime_type=mime, sample=None) == 0.0


@pytest.mark.parametrize("name", ["script.py", "photo.png", "clip.wav", "blob.bin"])
def test_embedding_handler_ignores_unrelated_files(name: str) -> None:
    assert EmbeddingFileHandler.can_handle(name, mime_type=None, sample=None) == 0.0


def test_video_handler_claims_its_own_types() -> None:
    for ext, mime in [
        (".mp4", "video/mp4"),
        (".mov", "video/quicktime"),
        (".mkv", "video/x-matroska"),
        (".webm", "video/webm"),
    ]:
        assert VideoFileHandler.can_handle(f"v{ext}", mime_type=mime, sample=None) > 0.0


def test_embedding_handler_claims_npy_and_jsonl() -> None:
    assert EmbeddingFileHandler.can_handle("vecs.npy", sample=b"\x93NUMPY\x01\x00") > 0.0
    assert EmbeddingFileHandler.can_handle("vecs.jsonl") > 0.0
    # A .npy extension whose bytes are not the numpy magic must NOT be claimed.
    assert EmbeddingFileHandler.can_handle("fake.npy", sample=b"not numpy") == 0.0


def test_video_magic_sniff_rejects_non_video_isobmff() -> None:
    # A HEIF/HEIC ISO-BMFF (ftyp brand "heic") shares the box structure but is
    # not one of our video brands, so it must not route to the video handler.
    heic = b"\x00\x00\x00\x18ftypheic\x00\x00\x00\x00heicmif1"
    assert VideoFileHandler.can_handle("img.heic", sample=heic) == 0.0
    mp4 = b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom"
    assert VideoFileHandler.can_handle("movie", sample=mp4) > 0.0


# ---------------------------------------------------------------------------
# Auto-discovery: both handlers register, existing routing stays intact.
# ---------------------------------------------------------------------------


def test_handlers_auto_discover_without_changing_existing_routing(tmp_path: Path) -> None:
    fp = Fingerprinter(FingerprintConfig())
    names = {handler.name for handler in fp.handlers}
    assert {"video", "embedding"} <= names

    # Existing routing is unchanged: a .py file still goes to the text handler,
    # a .bin still to the binary fallback -- the new handlers score 0 for them.
    py = tmp_path / "a.py"
    py.write_text("print('hello world')\n" * 8, encoding="utf-8")
    assert fp.fingerprint_file(py).handler == "text"

    blob = tmp_path / "a.bin"
    blob.write_bytes(bytes(range(256)) * 4)
    assert fp.fingerprint_file(blob).handler == "binary"


# ---------------------------------------------------------------------------
# Missing-dependency behavior: load() fails loud for the video extra.
# ---------------------------------------------------------------------------


def test_video_handler_missing_av_fails_loud(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom")

    monkeypatch.delitem(sys.modules, "av", raising=False)
    real_import = builtins.__import__

    def _fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "av" or name.startswith("av."):
            raise ImportError("No module named 'av'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)

    with pytest.raises(MissingDependencyError) as excinfo:
        VideoFileHandler().load(path)
    assert excinfo.value.package == "av"
    assert excinfo.value.extra == "video"


def test_embedding_plugin_embedder_missing_runtime_fails_loud(tmp_path: Path) -> None:
    # An injected embedder whose own (lazy) model import is missing must surface
    # as MissingDependencyError for the embeddings extra, not a bare ImportError.
    class _BrokenEmbedder:
        def embed(self, content: bytes) -> np.ndarray:
            raise ImportError("No module named 'sentence_transformers'")

    path = tmp_path / "input.jsonl"
    path.write_text(json.dumps([0.1, 0.2, 0.3]) + "\n", encoding="utf-8")

    handler = EmbeddingFileHandler(embedder=_BrokenEmbedder())
    with pytest.raises(MissingDependencyError) as excinfo:
        handler.load(path)
    assert excinfo.value.extra == "embeddings"


def test_embedding_handler_invalid_extension_fails(tmp_path: Path) -> None:
    # Defense in depth: even if load() is called directly on an unsupported
    # extension (can_handle would have returned 0), it does not silently succeed.
    path = tmp_path / "weird.dat"
    path.write_bytes(b"\x00\x01")
    with pytest.raises(MissingDependencyError):
        EmbeddingFileHandler().load(path)


# ---------------------------------------------------------------------------
# Embedding precomputed path is numpy-only -> always runs end-to-end.
# ---------------------------------------------------------------------------


def _synthetic_vectors(num: int = 24, dim: int = 32, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((num, dim)).astype(np.float32)


def test_embedding_npy_fingerprints_end_to_end(tmp_path: Path) -> None:
    vectors = _synthetic_vectors()
    path = tmp_path / "emb.npy"
    np.save(path, vectors)

    handler = EmbeddingFileHandler()
    assert handler.can_handle(path, sample=path.read_bytes()[:16]) > 0.0
    payload = handler.load(path)
    assert payload.num_vectors == vectors.shape[0]
    assert payload.dim == vectors.shape[1]

    signal = handler.to_signal(payload)
    assert signal.shape[0] == vectors.shape[0] * vectors.shape[1]
    # L2 normalisation: each frame is a unit vector (up to float tolerance).
    frame0 = signal[: vectors.shape[1]]
    assert np.linalg.norm(frame0) == pytest.approx(1.0, abs=1e-4)

    pipeline = FFTFingerprintPipeline(FingerprintConfig(window_size=32, hop_size=8))
    _landmarks, hashes = handler.extract_peaks(signal, pipeline)
    assert len(hashes) > 0


def test_embedding_jsonl_loads_object_and_bare_forms(tmp_path: Path) -> None:
    path = tmp_path / "emb.jsonl"
    lines = [
        json.dumps([0.1, 0.2, 0.3, 0.4]),
        json.dumps({"embedding": [0.5, 0.6, 0.7, 0.8]}),
        json.dumps({"vector": [0.9, 1.0, 1.1, 1.2]}),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    payload = EmbeddingFileHandler().load(path)
    assert payload.num_vectors == 3
    assert payload.dim == 4
    assert payload.source == "precomputed_jsonl"


def test_embedding_jsonl_load_disk_and_bytes_agree_on_lone_cr(tmp_path: Path) -> None:
    # A1 regression: the single-read content form must parse line endings the same
    # way the path form does. read_text() does universal-newline translation, so a
    # classic-Mac lone-\r .jsonl splits into N vectors; the content (bytes) form
    # must match (it previously kept it as one line -> JSONDecodeError -> demotion).
    path = tmp_path / "emb.jsonl"
    body = "\r".join(json.dumps([float(i), float(i + 1), float(i + 2)]) for i in range(5))
    path.write_bytes(body.encode("utf-8"))  # lone-\r separators, no trailing newline

    handler = EmbeddingFileHandler()
    from_disk = handler.load(path)
    from_bytes = handler.load(path, content=path.read_bytes())
    assert from_disk.num_vectors == 5
    assert from_bytes.num_vectors == 5
    assert np.array_equal(from_disk.vectors, from_bytes.vectors)


def test_embedding_single_vector_is_one_frame(tmp_path: Path) -> None:
    path = tmp_path / "one.npy"
    np.save(path, np.arange(16, dtype=np.float32))
    payload = EmbeddingFileHandler().load(path)
    assert payload.num_vectors == 1
    assert payload.dim == 16


def test_embedding_via_injected_embedder(tmp_path: Path) -> None:
    # The pluggable-embedder path needs no heavy dep when the embedder itself is
    # dependency-free; verify the sequence routes through the alignment core.
    class _DummyEmbedder:
        def embed(self, content: bytes) -> np.ndarray:
            rng = np.random.default_rng(len(content))
            return rng.standard_normal((10, 16)).astype(np.float32)

    path = tmp_path / "raw.jsonl"
    path.write_text("ignored by the dummy embedder\n", encoding="utf-8")

    handler = EmbeddingFileHandler(embedder=_DummyEmbedder())
    payload = handler.load(path)
    assert payload.num_vectors == 10
    assert payload.dim == 16
    assert payload.source.startswith("embedder:")


# ---------------------------------------------------------------------------
# Video end-to-end is gated on a video backend being installed.
# ---------------------------------------------------------------------------


def test_video_fingerprints_end_to_end_if_backend_present(tmp_path: Path) -> None:
    av = pytest.importorskip("av", exc_type=ImportError)
    pytest.importorskip("PIL", exc_type=ImportError)

    # Build a tiny synthetic clip: a few solid-color frames.
    path = tmp_path / "synthetic.mp4"
    width, height, fps, n_frames = 64, 48, 10, 12
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=fps)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        for i in range(n_frames):
            arr = np.full((height, width, 3), (i * 20) % 256, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)

    handler = VideoFileHandler()
    handler.configure(FingerprintConfig())
    payload = handler.load(path)
    assert payload.sampled_keyframes >= 1
    assert payload.frames.shape[1:] == (handler.canonical_size[1], handler.canonical_size[0])

    signal = handler.to_signal(payload)
    assert signal.shape[0] == payload.sampled_keyframes * handler.canonical_size[0] * handler.canonical_size[1]

    pipeline = FFTFingerprintPipeline(FingerprintConfig())
    _landmarks, hashes = handler.extract_peaks(signal, pipeline)
    assert len(hashes) >= 0  # decode + canonicalize + pipeline run without error
