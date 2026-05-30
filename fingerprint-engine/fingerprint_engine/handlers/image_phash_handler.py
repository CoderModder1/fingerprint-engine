"""OPT-IN perceptual-hash (pHash) image handler.

This is the SECOND of the two image handlers. The default
:class:`~fingerprint_engine.handlers.image_handler.ImageFileHandler` produces the
byte-identical raster 1D-signal constellation; this handler produces the 2D DCT
perceptual hash and is selected *only* when ``FingerprintConfig.image_mode ==
"phash"``. Splitting it out of the raster handler gives pHash fingerprints their
OWN handler name -- ``"image_phash"`` -- so they can be labelled distinctly and
given a stricter :class:`~fingerprint_engine.core.models.Calibration` cutoff via
``per_handler={"image_phash": ...}`` without touching the raster operating point.

Routing is mutually exclusive with the raster handler and gated on the active
mode (see :meth:`can_handle`): under the default ``image_mode="raster"`` this
handler declines every file (score ``0.0``) and the raster handler claims images
exactly as before, so the default path is unchanged. Only when ``image_mode``
is switched to ``"phash"`` does this handler claim ``image/*`` (and the raster
handler then declines), so there is never routing ambiguity between the two.
"""

from __future__ import annotations

import hashlib
import logging
from functools import lru_cache
from pathlib import Path

import numpy as np

from fingerprint_engine.core.fft_pipeline import FFTFingerprintPipeline
from fingerprint_engine.core.models import ConstellationHash, LandmarkPoint

from .image_handler import ImageFileHandler, ImagePayload

logger = logging.getLogger(__name__)

# Recommended per-handler accept cutoff for ``"image_phash"``. pHash recovers
# resize/crop/rotate/JPEG recall, but on smooth/synthetic images its global
# low-frequency descriptor also raises *impostor* confidence, so the global
# ``Calibration.default_min_confidence`` (0.05) leaks false accepts in phash mode
# (measured false-accept rate 1.0 at 0.05 on benchmarks/accuracy.py --mode hard).
# The accuracy harness shows a cutoff in the ~0.2-0.3 band absorbs that leak
# while keeping the recall win, so wire phash mode up with, e.g.::
#
#     Calibration(per_handler={IMAGE_PHASH_RECOMMENDED_CALIBRATION_KEY:
#                              IMAGE_PHASH_RECOMMENDED_MIN_CONFIDENCE})
#
# This does NOT change the global default cutoff; it is opt-in tuning for the
# (also opt-in) phash handler.
IMAGE_PHASH_RECOMMENDED_CALIBRATION_KEY = "image_phash"
IMAGE_PHASH_RECOMMENDED_MIN_CONFIDENCE = 0.25

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
# best handled with a stricter ``Calibration`` threshold for the phash handler.
_PHASH_BAND_BITS = 4
_PHASH_BANDS = _PHASH_BITS // _PHASH_BAND_BITS  # 16
# Namespacing salt so a phash band sub-code can never coincide with a raster
# constellation code for the same file id (different handlers/modes stay in
# disjoint hash regions).
_PHASH_NAMESPACE = b"fingerprint_engine.image.phash.v1"


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


class ImagePHashHandler(ImageFileHandler):
    """Perceptual-hash image handler, active only under ``image_mode="phash"``.

    It inherits the raster handler's loading/canonicalisation (decode -> grayscale
    -> canonical 256x256 grid -> normalised 1D signal) so a phash fingerprint is
    derived from the EXACT SAME canonical grid as raster; only the peak-extraction
    step differs (2D DCT perceptual hash instead of the FFT constellation). The
    class name is distinct (``"image_phash"``), so ``discover_handlers`` keeps it
    as a separate handler with its own calibration key, while the routing gate in
    :meth:`can_handle` makes the two handlers mutually exclusive.
    """

    name = "image_phash"
    # Same priority as the raster handler: routing is decided by the mode gate in
    # ``can_handle`` (exactly one of the two ever returns a non-zero score for a
    # given config), not by priority, so this never competes with raster.
    priority = ImageFileHandler.priority

    # The mode this handler SERVES. ``can_handle`` returns a score only when the
    # active (configured) mode equals this; otherwise it declines.
    _SERVED_MODE = "phash"

    def __init__(self, image_mode: str = "raster") -> None:
        super().__init__(image_mode)

    def can_handle(  # type: ignore[override]
        self,
        path: str | Path,
        mime_type: str | None = None,
        sample: bytes | None = None,
    ) -> float:
        """Claim images only when the active mode is ``"phash"``.

        Overridden as an INSTANCE method (the base ``can_handle`` is a
        classmethod, but ``_rank_handlers`` calls it on instances) so it can read
        the per-instance ``image_mode`` that ``configure`` set from the active
        config. Under the default ``"raster"`` mode this returns ``0.0`` for every
        file, so the phash handler never participates in routing and the raster
        handler's claim -- and therefore the whole default fingerprint path -- is
        unchanged. The underlying image scoring reuses the raster handler's shared
        ``_image_score`` classmethod (the identical scoring body), so when phash
        mode IS active this handler claims exactly the files raster would have.
        """

        if self.image_mode != self._SERVED_MODE:
            return 0.0
        return self._image_score(path, mime_type, sample)

    def extract_peaks(
        self,
        signal: np.ndarray,
        pipeline: FFTFingerprintPipeline,
    ) -> tuple[list[LandmarkPoint], list[ConstellationHash]]:
        """Derive the 2D DCT perceptual hash and emit one hash per pHash band.

        This is a SELF-CONTAINED alternate match path: it ignores the FFT
        pipeline, derives the 2D DCT perceptual hash from the canonical grayscale
        grid recovered from ``signal`` (the same canonical grid the raster
        handler builds), and emits one ``ConstellationHash`` per pHash band.

        Each band sub-code is emitted at a FIXED ``time_offset`` of 0 (and
        identical anchor/target times), so every band of a query and its match
        line up in the SAME offset histogram bin: the existing offset-alignment
        search then scores a candidate by how many bands (i.e. how much of the
        pHash) collide -- a Hamming-distance match with no change to the index or
        the search code. ``confidence`` stays calibrated because each file emits
        exactly ``_PHASH_BANDS`` band hashes, so a perfect match aligns all of
        them (confidence -> 1.0) and an unrelated image shares few or none.
        """

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
        return {
            "width": payload.width,
            "height": payload.height,
            "original_width": payload.original_size[0],
            "original_height": payload.original_size[1],
            "mode": payload.mode,
            "image_mode": self.image_mode,
            "signal_strategy": "canonical_256x256_dct_phash",
        }
