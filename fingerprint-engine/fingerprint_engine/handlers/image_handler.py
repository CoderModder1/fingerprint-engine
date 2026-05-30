"""Handler for raster images using grayscale intensity signals."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from fingerprint_engine.core.exceptions import MissingDependencyError
from fingerprint_engine.core.fft_pipeline import FFTFingerprintPipeline
from fingerprint_engine.core.models import ConstellationHash, FingerprintConfig, LandmarkPoint

from .base import FileHandler

logger = logging.getLogger(__name__)

# OPT-IN perceptual-hash ("phash") geometry. The 2D DCT is taken over the
# canonical grayscale grid and its top-left ``_PHASH_DCT_SIZE`` x
# ``_PHASH_DCT_SIZE`` low-frequency block (excluding the DC term) drives a
# ``_PHASH_BITS``-bit hash -- the classic pHash recipe (low frequencies are what
# survive resize / JPEG / a small crop or rotation; high frequencies are exactly
# the detail those operations destroy). 8x8 over the 256x256 canonical grid
# yields the standard 64-bit pHash.
_PHASH_DCT_SIZE = 8
_PHASH_BITS = _PHASH_DCT_SIZE * _PHASH_DCT_SIZE  # 64
# The 64-bit code is split into ``_PHASH_BITS // _PHASH_BAND_BITS`` fixed-width
# bands; each band becomes one searchable sub-code tagged with its band index, so
# two pHashes that differ by a few bits still share most bands and collide on the
# existing offset-histogram search. A single flipped bit invalidates exactly the
# ONE band that contains it, so NARROWER bands (more of them) lose a smaller
# fraction of bands per flip -> higher near-duplicate recall and higher
# true-match confidence; WIDER bands collide less by chance -> better impostor
# separation. Measured on the accuracy harness across both a smooth-gradient and
# a high-detail corpus, 4-bit bands (16 sub-codes) gave the best resize/JPEG
# recall and confidence at a tolerable separation cost, so they are the default.
# This is band-LSH over the Hamming ball, reusing the constellation index
# unchanged; the residual impostor-separation cost is a documented phash caveat
# best handled with a stricter ``Calibration`` threshold for phash mode.
_PHASH_BAND_BITS = 4
_PHASH_BANDS = _PHASH_BITS // _PHASH_BAND_BITS  # 16
# Namespacing salt so a phash band sub-code can never coincide with a raster
# constellation code for the same file id (different handlers/modes stay in
# disjoint hash regions).
_PHASH_NAMESPACE = b"fingerprint_engine.image.phash.v1"


@dataclass(frozen=True)
class ImagePayload:
    pixels: np.ndarray
    width: int
    height: int
    mode: str
    original_size: tuple[int, int] = (0, 0)


@lru_cache(maxsize=4)
def _dct_matrix(n: int) -> np.ndarray:
    """Orthonormal type-II DCT basis as an ``n x n`` matrix (numpy-only).

    ``X = D @ x`` gives the 1D DCT-II of ``x``; ``D @ M @ D.T`` is the separable
    2D DCT of ``M``. scipy/cv2 are deliberately avoided -- only the basis matrix
    is needed and it is tiny and cached. Computed in float64 for numerical
    stability, then the caller works in float64 so the bit decisions are
    deterministic across platforms.
    """

    k = np.arange(n).reshape(-1, 1)
    i = np.arange(n).reshape(1, -1)
    basis = np.cos(np.pi * (2 * i + 1) * k / (2 * n))
    basis *= np.sqrt(2.0 / n)
    basis[0, :] *= np.sqrt(0.5)
    return basis


def compute_phash_bits(pixels: np.ndarray) -> np.ndarray:
    """Return the ``_PHASH_BITS``-length boolean pHash of a grayscale grid.

    Standard DCT pHash: take the separable 2D DCT of the canonical grayscale
    grid, keep the top-left ``_PHASH_DCT_SIZE`` x ``_PHASH_DCT_SIZE`` low-frequency
    block, drop the DC (0, 0) term (it only encodes overall brightness, which a
    re-encode/exposure shift moves), and threshold each remaining coefficient
    against the median of the block. The result is a flat bit vector in
    row-major order over the DCT block. Low-frequency coefficients are what
    survive resize, JPEG, a modest crop, and a small rotation -- which is exactly
    why this is more geometry-robust than the row-major raster signal.
    """

    grid = np.asarray(pixels, dtype=np.float64)
    rows, cols = grid.shape
    basis_rows = _dct_matrix(rows)
    basis_cols = _dct_matrix(cols)
    dct = basis_rows @ grid @ basis_cols.T
    block = dct[:_PHASH_DCT_SIZE, :_PHASH_DCT_SIZE].reshape(-1)
    # Median over the block EXCLUDING the DC term, matching the classic recipe.
    median = float(np.median(block[1:]))
    bits = block > median
    bits[0] = False  # DC carries no perceptual structure; pin it for stability.
    return bits


def phash_band_codes(bits: np.ndarray) -> list[int]:
    """Split a pHash bit vector into position-tagged band sub-codes.

    Each consecutive ``_PHASH_BAND_BITS``-bit slice of the hash is packed into an
    integer and salted with (namespace, band index, band value) via blake2b, so:

    * a band's code depends on its POSITION, so band ``j`` of one image only ever
      collides with band ``j`` of another (no cross-band aliasing);
    * the codes live in a salted region disjoint from raster constellation codes;
    * two images whose pHashes differ in only a few bits share every band that
      contains none of the flipped bits, so they collide on most of their
      ``_PHASH_BANDS`` sub-codes -- a Hamming-ball match expressed as ordinary
      index postings, with no change to the index or search code.

    Returns one 64-bit code per band, in band order.
    """

    flat = np.asarray(bits, dtype=bool).reshape(-1)
    codes: list[int] = []
    for band in range(_PHASH_BANDS):
        start = band * _PHASH_BAND_BITS
        chunk = flat[start : start + _PHASH_BAND_BITS]
        value = 0
        for bit in chunk:
            value = (value << 1) | int(bit)
        payload = (
            _PHASH_NAMESPACE
            + band.to_bytes(2, "big")
            + value.to_bytes(8, "big")
        )
        digest = hashlib.blake2b(payload, digest_size=8).digest()
        codes.append(int.from_bytes(digest, "big", signed=False))
    return codes


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

    def __init__(self, image_mode: str = "raster") -> None:
        # ``"raster"`` (the default, and what the no-arg discovery uses) is the
        # byte-identical 1D-signal path; ``"phash"`` switches to the opt-in 2D
        # DCT perceptual-hash path. ``configure`` re-reads this from the active
        # FingerprintConfig so the global/CLI ``image_mode`` reaches the handler
        # the Fingerprinter instantiates with no arguments.
        if image_mode not in ("raster", "phash"):
            raise ValueError("image_mode must be 'raster' (default/off) or 'phash'")
        self.image_mode = image_mode

    def configure(self, config: FingerprintConfig) -> None:
        # Honor the configured image mode so FingerprintConfig.image_mode (and
        # any future --image-mode flag) reaches the discovered handler. Default
        # ("raster") leaves the byte-identical raster path in force.
        self.image_mode = config.image_mode

    @classmethod
    def can_handle(
        cls,
        path: str | Path,
        mime_type: str | None = None,
        sample: bytes | None = None,
    ) -> float:
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
            original_size = (int(image.width), int(image.height))
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

    def extract_peaks(
        self,
        signal: np.ndarray,
        pipeline: FFTFingerprintPipeline,
    ) -> tuple[list[LandmarkPoint], list[ConstellationHash]]:
        """Produce searchable hashes for the active image mode.

        ``"raster"`` (the default) defers entirely to the base implementation --
        the shared FFT pipeline over the 1D signal -- so the landmarks and
        constellation hashes are BYTE-IDENTICAL to before this handler grew an
        image mode. ``"phash"`` is a SELF-CONTAINED alternate match path: it
        ignores the FFT pipeline, derives the 2D DCT perceptual hash from the
        canonical grayscale grid recovered from ``signal``, and emits one
        ``ConstellationHash`` per pHash band.

        Each band sub-code is emitted at a FIXED ``time_offset`` of 0 (and
        identical anchor/target times), so every band of a query and its match
        line up in the SAME offset histogram bin: the existing offset-alignment
        search then scores a candidate by how many bands (i.e. how much of the
        pHash) collide -- a Hamming-distance match with no change to the index or
        the search code. ``confidence`` stays calibrated because each file emits
        exactly ``_PHASH_BANDS`` band hashes, so a perfect match aligns all of
        them (confidence -> 1.0) and an unrelated image shares few or none.
        """

        if self.image_mode != "phash":
            return super().extract_peaks(signal, pipeline)

        rows, cols = self.canonical_size[1], self.canonical_size[0]
        # ``signal`` is the (de-normalised) flattened canonical grid; recover it.
        grid = (np.asarray(signal, dtype=np.float64).reshape(rows, cols) * 127.5) + 127.5
        bits = compute_phash_bits(grid)
        codes = phash_band_codes(bits)
        hashes = [
            ConstellationHash(
                hash_code=code,
                time_offset=0,
                anchor_time=0,
                target_time=0,
                freq1=band,
                freq2=band,
                delta_t=0,
            )
            for band, code in enumerate(codes)
        ]
        hashes.sort(key=lambda item: (item.time_offset, item.hash_code, item.freq1, item.freq2))
        # pHash is a global descriptor, not a constellation of spatial peaks, so
        # there are no landmark points to report.
        return [], hashes

    def metadata(self, payload: ImagePayload) -> dict[str, object]:
        strategy = (
            "canonical_256x256_dct_phash"
            if self.image_mode == "phash"
            else "canonical_256x256_grayscale_intensity"
        )
        return {
            "width": payload.width,
            "height": payload.height,
            "original_width": payload.original_size[0],
            "original_height": payload.original_size[1],
            "mode": payload.mode,
            "image_mode": self.image_mode,
            "signal_strategy": strategy,
        }
