"""Handler that fingerprints a SEQUENCE of dense embedding vectors.

Heavy-dependency skeleton (item 3). It accepts *precomputed* dense vectors in a
numpy-only on-disk form (``.npy`` array, or ``.jsonl`` of per-line vectors), OR
a pluggable :class:`Embedder` that turns raw content into a vector sequence. The
precomputed path needs nothing beyond numpy (the one core dep), so it always
works here; any real embedder (e.g. sentence-transformers) is imported LAZILY
and raises :class:`MissingDependencyError` for the ``embeddings`` extra when
absent -- importing :mod:`fingerprint_engine` never pulls in a model runtime.

Design -- how dense vectors map onto the constellation / alignment model
------------------------------------------------------------------------
A dense embedding is a point in R^d; the engine's index is built for a 1D
*signal* that the FFT pipeline turns into spectro-temporal landmarks. We bridge
the two by treating an ORDERED LIST of vectors as a time series:

* Input is a ``(num_vectors, d)`` matrix -- one vector per "frame" in sequence
  order (e.g. successive sentences/chunks of a document, successive windows of a
  recording, successive shots of a video already embedded upstream).
* Each row is L2-normalised so vector MAGNITUDE (which carries no semantic
  direction information and varies by model/temperature) cannot dominate; only
  the DIRECTION survives, which is what cosine-similarity-style embeddings
  encode.
* The normalised rows are concatenated in order into one long 1D signal:
  ``signal = [v0_0 .. v0_{d-1}, v1_0 .. v1_{d-1}, ...]``. Because every frame is
  exactly ``d`` samples wide, a per-handler FIXED FFT window aligned to ``d``
  keeps each vector on the SAME time grid across files, so a shared run of
  embeddings (a quoted passage, a repeated segment) produces shared
  constellation hashes and aligns on the existing offset histogram -- no change
  to the index or search code.

This is deliberately a SEQUENCE / near-duplicate-of-a-stream matcher, not a
single-vector ANN nearest-neighbour search (which is a different data structure
entirely); it reuses the constellation core to answer "do these two embedding
streams share an aligned sub-sequence?". A single bare vector is the degenerate
1-frame case.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from fingerprint_engine.core.exceptions import MissingDependencyError
from fingerprint_engine.core.models import FingerprintConfig

from .base import FileHandler

logger = logging.getLogger(__name__)

# A fixed per-frame width is unknown until load() (it is the embedding
# dimensionality ``d``), so the handler sets its FFT window from the loaded
# payload via ``configure``-time defaults; see ``to_signal`` / the window note.
# These bound a pathological input so the concatenated signal cannot explode.
_DEFAULT_MAX_VECTORS = 4096
_DEFAULT_MAX_DIM = 8192


@runtime_checkable
class Embedder(Protocol):
    """Pluggable text/content -> vector-sequence encoder.

    Any object with this shape can be injected (dependency-free). The handler
    itself never imports a model; a caller wires a concrete embedder in. The
    return is a ``(num_vectors, d)`` float array.
    """

    def embed(self, content: bytes) -> np.ndarray: ...  # noqa: E704 - Protocol stub


@dataclass(frozen=True)
class EmbeddingPayload:
    """An ordered, L2-normalised stack of dense vectors.

    ``vectors`` has shape ``(num_vectors, dim)``; ``dim`` is the per-frame FFT
    window the signal is fingerprinted with so each vector stays one frame.
    """

    vectors: np.ndarray
    num_vectors: int
    dim: int
    source: str


class EmbeddingFileHandler(FileHandler):
    name = "embedding"
    # Low priority and narrow routing: this only claims explicit embedding
    # artifacts (``.npy`` / ``.npz`` / ``.jsonl`` of vectors), never general
    # files. ``.npy``/``.jsonl`` are otherwise unclaimed by other handlers, so
    # adding this does not change any existing file's routing.
    priority = 30
    supported_extensions = {".npy", ".npz", ".jsonl"}
    # No MIME prefixes/types on purpose: embedding artifacts have no reliable
    # registered MIME type, and claiming a broad prefix would mis-route files.

    def __init__(
        self,
        embedder: Embedder | None = None,
        max_vectors: int | None = None,
        max_dim: int | None = None,
    ) -> None:
        # No-arg discovery construction yields the precomputed-vector handler
        # (embedder=None). A caller can inject a concrete Embedder for the
        # encode-on-load path.
        self.embedder = embedder
        self.max_vectors = int(max_vectors if max_vectors is not None else _DEFAULT_MAX_VECTORS)
        self.max_dim = int(max_dim if max_dim is not None else _DEFAULT_MAX_DIM)
        if self.max_vectors <= 0:
            raise ValueError("max_vectors must be positive")
        if self.max_dim <= 0:
            raise ValueError("max_dim must be positive")
        # Per-frame FFT window == embedding dimensionality, learned at load().
        self._signal_window: int | None = None

    def configure(self, config: FingerprintConfig) -> None:
        # Forward-compatible config wiring; ``getattr`` keeps the default path
        # unchanged if these fields are not present on the config yet.
        self.max_vectors = int(getattr(config, "embedding_max_vectors", self.max_vectors))
        self.max_dim = int(getattr(config, "embedding_max_dim", self.max_dim))

    @classmethod
    def can_handle(
        cls,
        path: str | Path,
        mime_type: str | None = None,
        sample: bytes | None = None,
    ) -> float:
        suffix = Path(path).suffix.lower()
        if suffix not in cls.supported_extensions:
            # Hard 0 for everything else -- never claim arbitrary content even if
            # an embedder is wired in; the artifact extension is the only signal.
            return 0.0
        # ``.npy`` carries a stable magic header; confirm it when we have a
        # sample so a mislabeled file falls through instead of failing in load().
        if suffix in {".npy", ".npz"}:
            if sample is not None and not sample.startswith(b"\x93NUMPY") and not sample.startswith(b"PK"):
                return 0.0
            return 0.80
        return 0.70  # .jsonl

    def load(self, path: str | Path) -> EmbeddingPayload:
        source = Path(path)
        suffix = source.suffix.lower()

        if self.embedder is not None:
            vectors = self._embed_with_plugin(source)
            origin = f"embedder:{type(self.embedder).__name__}"
        elif suffix in {".npy", ".npz"}:
            vectors = self._load_npy(source)
            origin = "precomputed_npy"
        elif suffix == ".jsonl":
            vectors = self._load_jsonl(source)
            origin = "precomputed_jsonl"
        else:  # pragma: no cover - guarded by can_handle
            raise MissingDependencyError(
                f"unsupported embedding artifact {suffix!r}",
                extra="embeddings",
            )

        matrix = self._sanitize(vectors)
        num_vectors, dim = matrix.shape
        self._signal_window = dim
        return EmbeddingPayload(vectors=matrix, num_vectors=num_vectors, dim=dim, source=origin)

    def _embed_with_plugin(self, path: Path) -> np.ndarray:
        """Run the injected embedder, lazily importing nothing ourselves.

        The embedder may itself lazily import a heavy runtime (e.g.
        sentence-transformers). If that import fails the embedder should surface
        it; we wrap a bare ImportError into MissingDependencyError for the
        ``embeddings`` extra so the failure mode matches the other handlers.
        """

        try:
            assert self.embedder is not None
            result = self.embedder.embed(self.read_bytes(path))
        except ImportError as exc:
            raise MissingDependencyError(
                "the configured embedder requires an optional model runtime; install with "
                "'pip install \"fingerprint-engine[embeddings]\"'",
                extra="embeddings",
            ) from exc
        return np.asarray(result, dtype=np.float64)

    @staticmethod
    def _load_npy(path: Path) -> np.ndarray:
        # numpy is a core dependency, so the precomputed path needs no extra.
        # ``allow_pickle=False`` keeps deserialization safe on untrusted inputs.
        loaded = np.load(path, allow_pickle=False)
        if isinstance(loaded, np.lib.npyio.NpzFile):
            try:
                array = loaded[loaded.files[0]]
            finally:
                loaded.close()
            return np.asarray(array, dtype=np.float64)
        return np.asarray(loaded, dtype=np.float64)

    @staticmethod
    def _load_jsonl(path: Path) -> np.ndarray:
        rows: list[list[float]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                value = json.loads(line)
                # Accept a bare list, or an object with an "embedding"/"vector" key.
                if isinstance(value, dict):
                    value = value.get("embedding") or value.get("vector")
                if not isinstance(value, list):
                    raise ValueError("each .jsonl line must be a vector list or an {embedding|vector: [...]} object")
                rows.append([float(x) for x in value])
        if not rows:
            raise ValueError("no vectors found in .jsonl embedding file")
        return np.asarray(rows, dtype=np.float64)

    def _sanitize(self, vectors: np.ndarray) -> np.ndarray:
        """Coerce to a 2D ``(n, d)`` stack, bound it, and L2-normalise rows."""

        array = np.asarray(vectors, dtype=np.float64)
        if array.ndim == 1:
            array = array.reshape(1, -1)  # a single bare vector -> 1 frame
        if array.ndim != 2:
            raise ValueError(f"embedding input must be 1D or 2D, got shape {array.shape}")
        num_vectors, dim = array.shape
        if num_vectors == 0 or dim == 0:
            raise ValueError("embedding input is empty")
        if dim > self.max_dim:
            raise ValueError(f"embedding dimension {dim} exceeds max_dim {self.max_dim}")
        if num_vectors > self.max_vectors:
            array = array[: self.max_vectors]
        # L2-normalise each row so only direction (semantic content) drives the
        # signal; guard zero vectors to avoid div-by-zero / NaN.
        norms = np.linalg.norm(array, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        array = array / norms
        return np.nan_to_num(array.astype(np.float32), nan=0.0)

    def to_signal(self, payload: EmbeddingPayload) -> np.ndarray:
        # Row-major flatten preserves vector order: frame i occupies samples
        # [i*d, (i+1)*d). The fixed-window note in the module docstring explains
        # why a per-frame window of width ``d`` keeps each vector one frame.
        if payload.vectors.size == 0:
            return np.zeros(1, dtype=np.float32)
        return np.asarray(payload.vectors, dtype=np.float32).reshape(-1)

    def metadata(self, payload: EmbeddingPayload) -> dict[str, object]:
        return {
            "num_vectors": payload.num_vectors,
            "embedding_dim": payload.dim,
            "source": payload.source,
            "signal_strategy": "l2_normalized_vector_sequence",
        }
