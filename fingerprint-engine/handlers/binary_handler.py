"""Fallback handler for raw binary and unknown files."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from core.fft_pipeline import FFTFingerprintPipeline

from .base import FileHandler


class BinaryFileHandler(FileHandler):
    name = "binary"
    priority = -100

    @classmethod
    def can_handle(
        cls,
        path: str | Path,
        mime_type: str | None = None,
        sample: bytes | None = None,
    ) -> float:
        return 0.05

    def load(self, path: str | Path) -> bytes:
        return self.read_bytes(path)

    def to_signal(self, payload: bytes) -> np.ndarray:
        pipeline = FFTFingerprintPipeline()
        return pipeline.bytes_to_signal(payload)

    def metadata(self, payload: bytes) -> dict[str, object]:
        return {
            "byte_length": len(payload),
            "signal_strategy": "normalized_raw_bytes",
        }
