"""Abstract file handler plugin interface."""

from __future__ import annotations

import mimetypes
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

from fingerprint_engine.core.fft_pipeline import FFTFingerprintPipeline
from fingerprint_engine.core.models import ConstellationHash, FingerprintConfig, LandmarkPoint


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
    # Optional per-handler multi-resolution window bank, applied under the default
    # config (and only when no global ``FingerprintConfig.window_bank`` is set).
    # When present, the handler fingerprints the signal once per window in the
    # bank, folding the window size into each hash so window-w codes only collide
    # with window-w codes -- recovering the cross-length / excerpt matching a
    # single fixed window misses, at ~N x the postings. Audio sets this so
    # excerpt/clip matching works by default. None -> single-window behaviour.
    default_window_bank: tuple[int, ...] | None = None

    @classmethod
    def can_handle(
        cls,
        path: str | Path,
        mime_type: str | None = None,
        sample: bytes | None = None,
    ) -> float:
        """Return a confidence score between 0 and 1.

        Every candidate signal (extension, exact MIME, MIME prefix) is scored
        and the *strongest* one wins. Returning the first match in priority
        order would let a weak signal (e.g. an extension match, 0.75) mask a
        stronger one (an exact MIME match, 0.80) on the same file, so the MAX
        is taken instead. 0.0 means no evidence at all.
        """

        suffix = Path(path).suffix.lower()
        score = 0.0
        if suffix in cls.supported_extensions:
            score = max(score, 0.75)
        if mime_type and mime_type in cls.supported_mime_types:
            score = max(score, 0.80)
        if mime_type and any(mime_type.startswith(prefix) for prefix in cls.supported_mime_prefixes):
            score = max(score, 0.70)
        return score

    @abstractmethod
    def load(self, path: str | Path, *, content: bytes | None = None) -> Any:
        """Load a file into handler-specific payload form.

        ``content`` is the file's already-read bytes when the caller has them.
        :class:`Fingerprinter` reads each file ONCE -- for its content hash and
        identity (``content_sha256``/``file_id``) -- and threads those exact
        bytes here, so the bytes that get fingerprinted are provably the same
        bytes the stored identity describes: no second disk read, and no
        time-of-check/time-of-use window where a concurrent writer could make the
        fingerprinted bytes diverge from the recorded digest. ``None`` (the
        direct/legacy call form) means read from ``path``.
        """

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

    def configure(self, config: FingerprintConfig) -> None:  # noqa: B027 - intentional optional no-op hook
        """Apply config-derived per-handler settings. Default is a no-op.

        :class:`Fingerprinter` calls this once on each discovered handler so a
        handler can pull limits/tuning from the active config -- handlers are
        discovered and instantiated with no constructor arguments, so this is
        how a config value (e.g. the PDF page cap) reaches them. Override to
        read what the handler needs.
        """

    @staticmethod
    def sniff_mime(path: str | Path) -> str | None:
        mime_type, _encoding = mimetypes.guess_type(str(path))
        return mime_type

    @staticmethod
    def read_bytes(path: str | Path) -> bytes:
        return Path(path).read_bytes()
