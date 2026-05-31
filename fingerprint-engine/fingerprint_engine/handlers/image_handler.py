"""Handler for raster images using grayscale intensity signals.

This is the DEFAULT image handler. It produces the byte-identical raster
1D-signal constellation it always has. The opt-in 2D DCT perceptual-hash path
lives in a SEPARATE handler (:class:`~fingerprint_engine.handlers.image_phash_handler.ImagePHashHandler`,
name ``"image_phash"``) so that pHash fingerprints get their own handler name
and their own (stricter) calibration; see that module.

Routing between the two image handlers is mutually exclusive and gated on the
active ``FingerprintConfig.image_mode`` (see :meth:`can_handle`): under the
default ``"raster"`` this handler claims ``image/*`` and the phash handler
declines, so the default fingerprint path is unchanged; under ``"phash"`` this
handler declines and the phash handler claims them instead.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fingerprint_engine.core.exceptions import FileTooLargeError, MissingDependencyError
from fingerprint_engine.core.models import FingerprintConfig

from .base import FileHandler

logger = logging.getLogger(__name__)


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
    # Text/vector "image/*" types that the raster decoder (PIL grayscale resize)
    # cannot meaningfully handle. They share the ``image/`` MIME prefix but are
    # really XML/text, so they must NOT route here; excluding them lets them
    # fall through to the text/binary handlers instead of failing in load().
    excluded_mime_types = {
        "image/svg+xml",
        "image/svg",
    }
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

    # The mode this handler SERVES. The raster handler is the DEFAULT, so it
    # serves ``"raster"``; the phash subclass overrides this to ``"phash"``.
    _SERVED_MODE = "raster"

    def __init__(self, image_mode: str = "raster", max_image_pixels: int | None = None) -> None:
        # ``"raster"`` (the default, and what the no-arg discovery uses) is the
        # byte-identical 1D-signal path; ``"phash"`` switches routing to the
        # opt-in pHash handler. ``configure`` re-reads this from the active
        # FingerprintConfig so the global/CLI ``image_mode`` reaches the handler
        # the Fingerprinter instantiates with no arguments.
        if image_mode not in ("raster", "phash"):
            raise ValueError("image_mode must be 'raster' (default/off) or 'phash'")
        self.image_mode = image_mode
        # Decoded-pixel cap (decompression-bomb guard). None -> the config default
        # so the no-arg discovery construction picks it up; configure() re-reads it.
        if max_image_pixels is None:
            max_image_pixels = FingerprintConfig().max_image_pixels
        if max_image_pixels < 0:
            raise ValueError("max_image_pixels must be non-negative (0 = unlimited)")
        self.max_image_pixels = max_image_pixels

    def configure(self, config: FingerprintConfig) -> None:
        # Honor the configured image mode + pixel cap so FingerprintConfig values
        # reach the discovered handler (and, via inheritance, the phash handler).
        # Default ("raster") leaves the byte-identical raster path in force.
        self.image_mode = config.image_mode
        self.max_image_pixels = config.max_image_pixels

    def can_handle(  # type: ignore[override]
        self,
        path: str | Path,
        mime_type: str | None = None,
        sample: bytes | None = None,
    ) -> float:
        """Claim images only when the active mode is ``"raster"`` (the default).

        Overridden as an INSTANCE method (the base ``can_handle`` is a
        classmethod, but ``_rank_handlers`` calls it on instances) so it can read
        the per-instance ``image_mode`` that ``configure`` set from the active
        config. Under the default ``"raster"`` mode this is the only image
        handler that claims a file -- the phash handler declines -- so default
        routing, and the whole default fingerprint path, is unchanged. Under
        ``"phash"`` this handler declines (returns ``0.0``) and the phash handler
        claims the file instead, so the two are never both candidates.
        """

        if self.image_mode != self._SERVED_MODE:
            return 0.0
        return self._image_score(path, mime_type, sample)

    @classmethod
    def _image_score(
        cls,
        path: str | Path,
        mime_type: str | None = None,
        sample: bytes | None = None,
    ) -> float:
        """The mode-agnostic image confidence (shared by both image handlers).

        This is the original raster ``can_handle`` body, unchanged: it returns
        the same score the raster handler returned before the routing gate was
        added. Exposing it as a classmethod lets the phash subclass reuse the
        IDENTICAL image scoring once its own mode gate has passed, so when phash
        mode is active it claims exactly the files raster would have.
        """

        if mime_type and mime_type in cls.excluded_mime_types:
            return 0.0
        base_score = super().can_handle(path, mime_type, sample)
        if base_score:
            return base_score + 0.10
        if sample and (
            sample.startswith(b"\x89PNG")
            or sample.startswith(b"\xff\xd8\xff")
            or sample.startswith(b"GIF87a")
            or sample.startswith(b"GIF89a")
            or sample.startswith(b"BM")
            # WEBP: "RIFF" <4-byte size> "WEBP". Without this an extension-less /
            # MIME-less WEBP fell to the binary handler and would not match the
            # same WEBP fingerprinted with its extension. (.webp is already in
            # supported_extensions; this covers the content-sniff path.)
            or (sample.startswith(b"RIFF") and sample[8:12] == b"WEBP")
        ):
            return 0.90
        return 0.0

    def load(self, path: str | Path) -> ImagePayload:
        try:
            from PIL import Image
        except ImportError as exc:
            logger.warning(
                "missing optional dependency %s (extra %s) for image fingerprinting",
                "Pillow",
                "image",
            )
            raise MissingDependencyError(
                "Pillow is required for image fingerprinting; install with "
                "'pip install fingerprint_engine[image]'",
                package="Pillow",
                extra="image",
            ) from exc

        with Image.open(path) as image:
            mode = image.mode
            width, height = int(image.width), int(image.height)
            original_size = (width, height)
            # Decompression-bomb guard: Image.open is lazy (the size comes from the
            # header, no pixels decoded yet), so reject an over-cap image BEFORE
            # convert/resize pulls hundreds of megapixels into memory. A tiny but
            # highly-compressible file bypasses the on-disk max_file_size_bytes cap;
            # this bounds the DECODED footprint instead of trusting Pillow's
            # lenient default. 0 = unlimited (opt-out).
            if self.max_image_pixels and width * height > self.max_image_pixels:
                raise FileTooLargeError(
                    f"{Path(path).name}: decoded image is {width}x{height} = "
                    f"{width * height} pixels, exceeding max_image_pixels limit of "
                    f"{self.max_image_pixels}",
                    size=width * height,
                    limit=self.max_image_pixels,
                )
            resampling = getattr(Image, "Resampling", Image).LANCZOS
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
            "image_mode": self.image_mode,
            "signal_strategy": "canonical_256x256_grayscale_intensity",
        }
