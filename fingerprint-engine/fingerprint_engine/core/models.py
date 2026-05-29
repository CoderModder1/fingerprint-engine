"""Dataclasses used by the fingerprinting engine."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, order=True)
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


@dataclass(frozen=True)
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


@dataclass(frozen=True)
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
    """

    default_min_confidence: float = 0.05
    per_handler: dict[str, float] = field(default_factory=dict)

    def min_confidence(self, handler: str) -> float:
        return self.per_handler.get(handler, self.default_min_confidence)

    def accepts(self, handler: str, confidence: float) -> bool:
        return confidence >= self.min_confidence(handler)
