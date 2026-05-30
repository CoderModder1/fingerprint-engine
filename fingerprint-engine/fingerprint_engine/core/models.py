"""Dataclasses used by the fingerprinting engine."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

# Version of the HASH-DERIVATION format -- the rule set that turns a file's
# bytes into ``hash_code`` values. This is DISTINCT from the snapshot
# ``schema_version`` (which versions the *container* that serializes postings,
# see ``core/index.py``): two builds can share a snapshot schema yet derive
# incompatible hash codes. ``1`` is the default derivation shipped today.
#
# Matching is only valid between a query and an index built with the SAME
# format version: a hash code carries no meaning across formats, so a
# cross-format "match" is a false result, not a weak one. Flipping any
# HASH-CHANGING default (the constellation packing, the per-handler windows, a
# new canonical image transform, ...) MUST bump this constant AND require
# re-indexing existing corpora -- see VERSIONING.md.
#
# The opt-in, default-off hash-changing flags do not change this baseline
# (their default values leave the derivation byte-identical); instead, when one
# is *enabled*, ``effective_format_version`` reports a DIFFERENT version for
# that config, so an index built with such a flag is detectably incompatible
# with a default index without flipping any default. The offsets below are the
# additive per-flag bumps; they are deliberately distinct so a config that
# enables several flags lands on a value distinct from enabling any one alone.
FINGERPRINT_FORMAT_VERSION = 1

# Key under which the effective format version is recorded in
# ``Fingerprint.config`` (a metadata-only stamp; it is NOT a tuning parameter,
# is NOT consumed by the FFT pipeline, and never enters a hash payload).
FORMAT_VERSION_KEY = "fingerprint_format_version"

# Additive offsets applied to ``FINGERPRINT_FORMAT_VERSION`` when an opt-in
# hash-changing flag is enabled, so an index built with the flag records a
# version distinct from the default and from the other flags. These describe
# the *effective* derivation of a non-default config only; the default config
# always reports the bare ``FINGERPRINT_FORMAT_VERSION``.
_FORMAT_BUMP_FREQ_QUANTIZATION = 1000
_FORMAT_BUMP_WINDOW_BANK = 2000
_FORMAT_BUMP_IMAGE_PHASH = 4000


def effective_format_version(config: FingerprintConfig) -> int:
    """Return the hash-derivation format version a ``config`` records.

    A default :class:`FingerprintConfig` -- and any config whose hash-changing
    fields are all at their defaults -- reports :data:`FINGERPRINT_FORMAT_VERSION`
    unchanged, so the stamped version is byte-identical to today for every
    existing index. Enabling an opt-in HASH-CHANGING flag (``freq_quantization``
    > 1, a ``window_bank``, or ``image_mode == "phash"``) adds that flag's
    distinct offset, so a config that derives different hash codes reports a
    different version and an index built with it is *detectably* incompatible
    with a default index (see :func:`HashIndex.search`). This is the mechanism
    that makes a future default-flip a deliberate version bump rather than a
    silent corpus corruption; it changes no hash code and no ranking itself.
    """

    version = FINGERPRINT_FORMAT_VERSION
    if config.freq_quantization > 1:
        version += _FORMAT_BUMP_FREQ_QUANTIZATION
    if config.window_bank:
        version += _FORMAT_BUMP_WINDOW_BANK
    if config.image_mode == "phash":
        version += _FORMAT_BUMP_IMAGE_PHASH
    return version


@dataclass(frozen=True)
class FingerprintConfig:
    """Tunable parameters for the FFT-equivalent fingerprint pipeline."""

    window_size: int = 4096
    hop_size: int = 1024
    peak_threshold: float = 1.25
    peak_percentile: float = 90.0
    max_peaks_per_frame: int = 8
    constellation_fanout: int = 6
    min_delta_t: int = 1
    max_delta_t: int = 48
    hash_bits: int = 64
    # OPT-IN spectral-shift tolerance. The constellation hash is over the exact
    # (freq1, freq2, delta_t) tuple, so a one-bin spectral shift (e.g. a JPEG
    # re-encode or a small char edit nudging a peak) produces a different code
    # and a true match scatters. ``freq_quantization`` snaps each frequency bin
    # to a coarser grid of width ``freq_quantization`` (``bin // q``) BEFORE
    # hashing, so peaks within the same coarse band collide and survive the
    # shift. ``1`` (the default) is exact, current behaviour -- no quantization
    # is applied and hashes are byte-identical to before this flag existed.
    # Larger values trade fingerprint specificity (more collisions, lower
    # confidence separation) for shift tolerance.
    freq_quantization: int = 1
    # OPT-IN multi-resolution window bank (targets the cross-length and
    # audio-excerpt limitation). Matching needs the query and the reference to
    # share an *effective* window: a single fixed window misses cross-window /
    # cross-length cases (e.g. an audio excerpt re-normalises and shifts the
    # global time grid, so its single-window hashes do not collide with the
    # whole-file's -- audio-excerpt recall is ~0). When ``window_bank`` is set,
    # the pipeline fingerprints the signal once PER window in the bank, folding
    # the window size into each hash's derivation so window-w hashes only ever
    # collide with window-w hashes. A query fingerprinted with the same bank then
    # has, at every bank window, codes that can align with refs at that window --
    # so an excerpt that only aligns at a small window still finds its parent.
    # ``None`` (the default) is OFF: the single-window pipeline runs exactly as
    # before and the produced hashes are BYTE-IDENTICAL to before this flag
    # existed. A bank of N windows multiplies a file's posting count by roughly N
    # (one full constellation per window), so keep the bank small (<= 6 windows)
    # -- it is a storage/recall trade, not a free win. Each entry must be a
    # distinct window >= ``min_window_size``; the per-window hop preserves the
    # configured ``window_size``:``hop_size`` overlap ratio.
    window_bank: tuple[int, ...] | None = None
    max_window_bank_size: int = 6
    max_signal_samples: int = 2_000_000
    min_time_frames: int = 16
    min_window_size: int = 16
    # Resource limits for untrusted input (see SECURITY.md). A finite default
    # for max_file_size_bytes bounds the OOM vector from a hostile/huge file
    # while sitting far above any normal source/image/pdf/audio input. Use 0 to
    # opt out (unlimited). max_pdf_pages bounds how many PDF pages a handler will
    # decode; 0 means unlimited (enforcement lives in the PDF handler).
    max_file_size_bytes: int = 256 * 1024 * 1024  # 256 MiB; 0 = unlimited
    max_pdf_pages: int = 0  # 0 = unlimited
    # OPT-IN 2D image matching mode (targets the image resize/crop/rotate
    # robustness limitation). The default raster path flattens the canonical
    # 256x256 grayscale grid row-major into a single 1D signal; that couples
    # rows by wraparound and is fragile to any vertical shift (a crop or a small
    # rotation moves every row, so the README disclaims crop/rotate). ``"phash"``
    # instead derives a 2D DCT perceptual hash of the canonical grayscale image
    # and emits it as a bundle of position-tagged sub-codes so the EXISTING
    # offset-histogram index/search matches two images by how many sub-codes
    # (i.e. how much of the pHash) survive -- a Hamming-distance match in
    # disguise, far more resize/crop/rotate robust. ``"raster"`` (the default) is
    # OFF: the image handler produces the byte-identical 1D-signal constellation
    # hashes it always has, and only the image handler reads this field, so every
    # other content type is wholly unaffected. ``image_mode`` changes ONLY how
    # ``image/*`` files are fingerprinted; it is a per-handler routing switch, not
    # a pipeline parameter, so it is validated for membership only.
    image_mode: Literal["raster", "phash"] = "raster"

    def validate(self) -> None:
        if self.window_size < 8:
            raise ValueError("window_size must be at least 8")
        if self.hop_size < 1:
            raise ValueError("hop_size must be at least 1")
        if self.max_peaks_per_frame < 1:
            raise ValueError("max_peaks_per_frame must be at least 1")
        if self.constellation_fanout < 1:
            raise ValueError("constellation_fanout must be at least 1")
        if self.min_delta_t < 0:
            raise ValueError("min_delta_t must be non-negative")
        if self.max_delta_t < self.min_delta_t:
            raise ValueError("max_delta_t must be >= min_delta_t")
        if not 1 <= self.hash_bits <= 64:
            raise ValueError("hash_bits must be between 1 and 64")
        if self.freq_quantization < 1:
            raise ValueError("freq_quantization must be at least 1 (1 = off/exact)")
        if self.max_window_bank_size < 1:
            raise ValueError("max_window_bank_size must be at least 1")
        if self.window_bank is not None:
            bank = self.window_bank
            if not isinstance(bank, tuple):
                raise ValueError("window_bank must be a tuple of window sizes or None")
            if not bank:
                raise ValueError("window_bank must be non-empty when set (use None to disable)")
            if len(bank) > self.max_window_bank_size:
                raise ValueError(
                    f"window_bank has {len(bank)} windows; max_window_bank_size is "
                    f"{self.max_window_bank_size} (a bank of N windows ~N-folds postings)"
                )
            if len(set(bank)) != len(bank):
                raise ValueError("window_bank entries must be distinct")
            for window in bank:
                if not isinstance(window, int):
                    raise ValueError("window_bank entries must be ints")
                if window < self.min_window_size:
                    raise ValueError(
                        f"window_bank entry {window} is below min_window_size "
                        f"{self.min_window_size}"
                    )
                if window > self.max_signal_samples:
                    raise ValueError(
                        f"window_bank entry {window} exceeds max_signal_samples "
                        f"{self.max_signal_samples}"
                    )
        if self.max_signal_samples < self.window_size:
            raise ValueError("max_signal_samples must be >= window_size")
        if self.min_time_frames < 1:
            raise ValueError("min_time_frames must be at least 1")
        if self.min_window_size < 8:
            raise ValueError("min_window_size must be at least 8")
        if self.min_window_size > self.window_size:
            raise ValueError("min_window_size must be <= window_size")
        if not 0.0 <= self.peak_percentile <= 100.0:
            raise ValueError("peak_percentile must be between 0.0 and 100.0")
        if self.peak_threshold < 0.0:
            raise ValueError("peak_threshold must be non-negative")
        if self.max_file_size_bytes < 0:
            raise ValueError("max_file_size_bytes must be non-negative (0 = unlimited)")
        if self.max_pdf_pages < 0:
            raise ValueError("max_pdf_pages must be non-negative (0 = unlimited)")
        if self.image_mode not in ("raster", "phash"):
            raise ValueError("image_mode must be 'raster' (default/off) or 'phash'")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, order=True, slots=True)
class LandmarkPoint:
    """A peak in the spectrogram-like signal space."""

    time_index: int
    frequency_bin: int
    magnitude: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "time_index": self.time_index,
            "frequency_bin": self.frequency_bin,
            "magnitude": self.magnitude,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LandmarkPoint:
        return cls(
            time_index=int(data["time_index"]),
            frequency_bin=int(data["frequency_bin"]),
            magnitude=float(data["magnitude"]),
        )


@dataclass(frozen=True, slots=True)
class ConstellationHash:
    """A searchable code derived from a pair of landmark peaks."""

    hash_code: int
    time_offset: int
    anchor_time: int
    target_time: int
    freq1: int
    freq2: int
    delta_t: int

    def to_tuple(self) -> tuple[int, int]:
        return self.hash_code, self.time_offset

    def to_dict(self) -> dict[str, Any]:
        return {
            "hash_code": self.hash_code,
            "time_offset": self.time_offset,
            "anchor_time": self.anchor_time,
            "target_time": self.target_time,
            "freq1": self.freq1,
            "freq2": self.freq2,
            "delta_t": self.delta_t,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConstellationHash:
        return cls(
            hash_code=int(data["hash_code"]),
            time_offset=int(data["time_offset"]),
            anchor_time=int(data["anchor_time"]),
            target_time=int(data["target_time"]),
            freq1=int(data["freq1"]),
            freq2=int(data["freq2"]),
            delta_t=int(data["delta_t"]),
        )


@dataclass
class Fingerprint:
    """Fingerprint output for one file."""

    file_id: str
    path: str
    handler: str
    size_bytes: int
    content_sha256: str
    config: dict[str, Any]
    landmarks: list[LandmarkPoint] = field(default_factory=list)
    hashes: list[ConstellationHash] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def hash_count(self) -> int:
        return len(self.hashes)

    @property
    def landmark_count(self) -> int:
        return len(self.landmarks)

    @property
    def format_version(self) -> int:
        """The hash-derivation format version recorded on this fingerprint.

        Read from the :data:`FORMAT_VERSION_KEY` stamp the fingerprinter writes
        into :attr:`config`. A fingerprint produced or loaded WITHOUT the stamp
        (legacy data written before the field existed, or a snapshot-rebuilt
        fingerprint) is treated as the default :data:`FINGERPRINT_FORMAT_VERSION`,
        so absence is backward-compatible rather than a mismatch.
        """

        try:
            return int(self.config.get(FORMAT_VERSION_KEY, FINGERPRINT_FORMAT_VERSION))
        except (TypeError, ValueError):
            return FINGERPRINT_FORMAT_VERSION

    def hash_tuples(self) -> list[tuple[int, int]]:
        return [item.to_tuple() for item in self.hashes]

    def to_dict(self, include_landmarks: bool = True) -> dict[str, Any]:
        data = {
            "file_id": self.file_id,
            "path": self.path,
            "handler": self.handler,
            "size_bytes": self.size_bytes,
            "content_sha256": self.content_sha256,
            "config": self.config,
            "hashes": [item.to_dict() for item in self.hashes],
            "metadata": self.metadata,
        }
        if include_landmarks:
            data["landmarks"] = [item.to_dict() for item in self.landmarks]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Fingerprint:
        return cls(
            file_id=str(data["file_id"]),
            path=str(data["path"]),
            handler=str(data["handler"]),
            size_bytes=int(data["size_bytes"]),
            content_sha256=str(data["content_sha256"]),
            config=dict(data.get("config", {})),
            landmarks=[
                LandmarkPoint.from_dict(item)
                for item in data.get("landmarks", [])
            ],
            hashes=[
                ConstellationHash.from_dict(item)
                for item in data.get("hashes", [])
            ],
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True, slots=True)
class IndexPosting:
    """One occurrence of a constellation hash in an indexed file."""

    file_id: str
    hash_code: int
    time_offset: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "file_id": self.file_id,
            "hash_code": self.hash_code,
            "time_offset": self.time_offset,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IndexPosting:
        return cls(
            file_id=str(data["file_id"]),
            hash_code=int(data["hash_code"]),
            time_offset=int(data["time_offset"]),
        )


@dataclass(frozen=True)
class SearchResult:
    """Ranked match returned from a hash index."""

    file_id: str
    score: float
    aligned_votes: int
    total_votes: int
    unique_hashes: int
    offset: int
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "file_id": self.file_id,
            "score": self.score,
            "confidence": self.confidence,
            "aligned_votes": self.aligned_votes,
            "total_votes": self.total_votes,
            "unique_hashes": self.unique_hashes,
            "offset": self.offset,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class Calibration:
    """Per-handler minimum match confidence for accept/reject decisions.

    ``SearchResult.confidence`` (aligned votes / the smaller fingerprint's hash
    count) is already normalised to [0, 1] and comparable across handlers, so a
    single ``default_min_confidence`` usually suffices. ``per_handler`` overrides
    let a content type use a stricter or looser cutoff when warranted.

    ``offset_tolerance`` is an OPT-IN search-time setting (see
    :meth:`HashIndex.search`). ``0`` (the default) is OFF: the winning offset bin
    is the single exact-delta histogram peak, so search rankings are
    BYTE-IDENTICAL to behaviour before this field existed. When ``> 0`` the
    winning bin's vote count sums the +-tolerance neighbouring delta bins, which
    recovers recall on multi-edit near-duplicates whose votes otherwise fragment
    across adjacent delta bins. An explicit ``offset_tolerance`` passed to
    :meth:`HashIndex.search` overrides this field.
    """

    default_min_confidence: float = 0.05
    per_handler: dict[str, float] = field(default_factory=dict)
    offset_tolerance: int = 0

    def __post_init__(self) -> None:
        if self.offset_tolerance < 0:
            raise ValueError("offset_tolerance must be non-negative (0 = off/exact)")

    def min_confidence(self, handler: str) -> float:
        return self.per_handler.get(handler, self.default_min_confidence)

    def accepts(self, handler: str, confidence: float) -> bool:
        return confidence >= self.min_confidence(handler)
