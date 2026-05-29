"""Handler for raster images using grayscale intensity signals."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .base import FileHandler


@dataclass(frozen=True)
class ImagePayload:
    pixels: np.ndarray
    width: int
    height: int
    mode: str
    original_size: tuple[int, int] = (0, 0)


class ImageFileHandler(FileHandler):
    name = "image"
    priority = 60
    # Every image is resampled to this canonical grayscale grid before the
    # signal is built, so the same picture at different resolutions (and after
    # lossy re-encoding) maps to a comparable signal -- i.e. resolution-invariant
    # matching, the perceptual-hash approach. A flattened raw-pixel signal is
    # otherwise destroyed by any resize.
    canonical_size = (256, 256)
    supported_mime_prefixes = {"image/"}
    supported_extensions = {
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".bmp",
        ".tiff",
        ".tif",
        ".webp",
    }

    @classmethod
    def can_handle(
        cls,
        path: str | Path,
        mime_type: str | None = None,
        sample: bytes | None = None,
    ) -> float:
        base_score = super().can_handle(path, mime_type, sample)
        if base_score:
            return base_score + 0.10
        if sample and (
            sample.startswith(b"\x89PNG")
            or sample.startswith(b"\xff\xd8\xff")
            or sample.startswith(b"GIF87a")
            or sample.startswith(b"GIF89a")
            or sample.startswith(b"BM")
        ):
            return 0.90
        return 0.0

    def load(self, path: str | Path) -> ImagePayload:
        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("Pillow is required for image fingerprinting") from exc

        with Image.open(path) as image:
            mode = image.mode
            original_size = (int(image.width), int(image.height))
            resampling = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
            grayscale = image.convert("L").resize(self.canonical_size, resampling)
            pixels = np.asarray(grayscale, dtype=np.float32)
        width, height = self.canonical_size
        return ImagePayload(
            pixels=pixels,
            width=width,
            height=height,
            mode=mode,
            original_size=original_size,
        )

    def to_signal(self, payload: ImagePayload) -> np.ndarray:
        return (payload.pixels.reshape(-1) - 127.5) / 127.5

    def metadata(self, payload: ImagePayload) -> dict[str, object]:
        return {
            "width": payload.width,
            "height": payload.height,
            "original_width": payload.original_size[0],
            "original_height": payload.original_size[1],
            "mode": payload.mode,
            "signal_strategy": "canonical_256x256_grayscale_intensity",
        }
