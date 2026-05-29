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


class ImageFileHandler(FileHandler):
    name = "image"
    priority = 60
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
            grayscale = image.convert("L")
            width, height = grayscale.size
            max_pixels = 1_000_000
            if width * height > max_pixels:
                scale = (max_pixels / float(width * height)) ** 0.5
                new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
                resampling = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
                grayscale = grayscale.resize(new_size, resampling)
                width, height = grayscale.size
            pixels = np.asarray(grayscale, dtype=np.float32)
        return ImagePayload(pixels=pixels, width=width, height=height, mode=mode)

    def to_signal(self, payload: ImagePayload) -> np.ndarray:
        return (payload.pixels.reshape(-1) - 127.5) / 127.5

    def metadata(self, payload: ImagePayload) -> dict[str, object]:
        return {
            "width": payload.width,
            "height": payload.height,
            "mode": payload.mode,
            "signal_strategy": "flattened_grayscale_intensity",
        }
