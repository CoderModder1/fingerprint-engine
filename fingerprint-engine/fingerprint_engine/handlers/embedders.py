"""Concrete :class:`~fingerprint_engine.handlers.embedding_handler.Embedder`
implementations for the embedding handler's *encode-on-load* path.

The embedding handler's precomputed-vector path (``.npy`` / ``.npz`` / ``.jsonl``)
is numpy-only and always available. To fingerprint RAW text by encoding it at
load time, wire one of these encoders into
``EmbeddingFileHandler(embedder=...)``. Each encoder lazily imports its model
runtime via :func:`require_optional`, so importing :mod:`fingerprint_engine`
never pulls in a model and a core-only install is unaffected.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import require_optional

# The default static model: lightweight, distilled, and -- crucially -- runs
# with no torch dependency, so it loads in well under a second from cache.
DEFAULT_MODEL2VEC_MODEL = "minishlab/potion-base-8M"

# Loaded models are cached per name within a process so reuse (e.g. across a
# batch or a test module) does not reload the static model each call. The model
# object is third-party and untyped, so it is held as Any.
_STATIC_MODEL_CACHE: dict[str, Any] = {}


class Model2VecEmbedder:
    """:class:`Embedder` adapter over a model2vec ``StaticModel`` (no torch).

    ``embed(content) -> (n_chunks, d)`` decodes the bytes as UTF-8, splits the
    text into non-empty lines (one line == one chunk == one embedding vector ==
    one downstream FFT frame), and runs ``model.encode(chunks)``. The static
    distilled model is deterministic -- no sampling, no torch -- so identical
    text always yields byte-identical vectors, which is what makes the resulting
    fingerprint stable and reproducible across machines.

    Requires the ``embeddings`` extra::

        pip install "fingerprint-engine[embeddings]"

    model2vec is imported LAZILY in ``__init__`` (the first time an encoder is
    constructed), via :func:`require_optional`, so a missing model runtime fails
    loud with :class:`~fingerprint_engine.core.exceptions.MissingDependencyError`
    for the ``embeddings`` extra rather than being silently swallowed.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL2VEC_MODEL) -> None:
        self.model_name = model_name
        model: Any = _STATIC_MODEL_CACHE.get(model_name)
        if model is None:
            module = require_optional(
                "model2vec",
                package="model2vec",
                extra="embeddings",
                message=(
                    "model2vec is required for the encode-on-load embedding path; "
                    "install with 'pip install \"fingerprint-engine[embeddings]\"'"
                ),
            )
            model = module.StaticModel.from_pretrained(model_name)
            _STATIC_MODEL_CACHE[model_name] = model
        self._model = model

    def embed(self, content: bytes) -> np.ndarray:
        text = content.decode("utf-8")
        chunks = [line.strip() for line in text.splitlines() if line.strip()]
        if not chunks:  # never hand the model an empty batch
            chunks = [text]
        return np.asarray(self._model.encode(chunks), dtype=np.float64)
