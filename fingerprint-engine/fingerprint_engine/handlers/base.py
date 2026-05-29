"""Abstract file handler plugin interface."""

from __future__ import annotations

import mimetypes
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

from fingerprint_engine.core.fft_pipeline import FFTFingerprintPipeline
from fingerprint_engine.core.models import ConstellationHash, LandmarkPoint


class FileHandler(ABC):
    """Base class for pluggable file-type signal extractors."""

    name = "base"
    priority = 0
    supported_extensions: set[str] = set()
    supported_mime_types: set[str] = set()
    supported_mime_prefixes: set[str] = set()
    # Optional per-handler preferred FFT window/hop, applied under the default
    # config. A small *fixed* window keeps a content type's fingerprints
    # comparable across files of different lengths, so excerpts/truncations of
    # the same content still align on a shared time grid (a length-adaptive
    # window would shift that grid and break matching). None -> use config.
    default_signal_window: int | None = None
    default_signal_hop: int | None = None

    @classmethod
    def can_handle(
        cls,
        path: str | Path,
        mime_type: str | None = None,
        sample: bytes | None = None,
    ) -> float:
        """Return a confidence score between 0 and 1."""

        suffix = Path(path).suffix.lower()
        if suffix in cls.supported_extensions:
            return 0.75
        if mime_type and mime_type in cls.supported_mime_types:
            return 0.80
        if mime_type and any(mime_type.startswith(prefix) for prefix in cls.supported_mime_prefixes):
            return 0.70
        return 0.0

    @abstractmethod
    def load(self, path: str | Path) -> Any:
        """Load a file into handler-specific payload form."""

    @abstractmethod
    def to_signal(self, payload: Any) -> np.ndarray:
        """Convert payload into a 1D numeric signal."""

    def extract_peaks(
        self,
        signal: np.ndarray,
        pipeline: FFTFingerprintPipeline,
    ) -> tuple[list[LandmarkPoint], list[ConstellationHash]]:
        """Run the shared signal pipeline."""

        return pipeline.fingerprint_signal(signal)

    def metadata(self, payload: Any) -> dict[str, Any]:
        """Return optional metadata for the fingerprint and index."""

        return {}

    @staticmethod
    def sniff_mime(path: str | Path) -> str | None:
        mime_type, _encoding = mimetypes.guess_type(str(path))
        return mime_type

    @staticmethod
    def read_bytes(path: str | Path) -> bytes:
        return Path(path).read_bytes()
