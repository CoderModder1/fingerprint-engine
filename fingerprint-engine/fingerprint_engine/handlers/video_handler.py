"""Handler that fingerprints video as a sequence of canonical keyframes.

Heavy-dependency skeleton (item 3). The decode path is behind a LAZY import so
importing :mod:`fingerprint_engine` -- and discovering this handler -- never
requires a video library. When no video backend is installed, :meth:`load`
raises :class:`MissingDependencyError` pointing at the ``video`` extra, exactly
like the existing image/audio/pdf handlers, rather than silently demoting to the
binary fallback (which would produce an incomparable fingerprint).

Design (see the module-level constants and ``load``):

* Keyframes are sampled at a FIXED temporal cadence (one frame every
  ``keyframe_interval_seconds`` of wall-clock video time, derived from the
  container frame rate), NOT at codec I-frame boundaries. A fixed cadence keeps
  the keyframe grid comparable across re-encodes, trims, and different GOP
  structures, so an excerpt of a clip still lands on the same grid -- the same
  reasoning behind the fixed per-handler FFT window used elsewhere.
* Each sampled frame is reduced to the SAME canonical grid the image handler
  uses (256x256 grayscale, Lanczos) so a frame contributes a
  resolution-invariant, perceptual signal block. The canonicalization is reused
  verbatim from :class:`ImageFileHandler` (one code path, one definition of
  "canonical frame").
* The per-frame canonical grids are flattened and CONCATENATED, in time order,
  into one long 1D signal. The existing FFT constellation pipeline then
  fingerprints the video as a keyframe *sequence*: temporally local frame
  structure becomes the spectro-temporal landmarks, so two videos that share a
  run of frames share constellation hashes and align on the offset histogram --
  no change to the index or search code.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fingerprint_engine.core.exceptions import MissingDependencyError
from fingerprint_engine.core.models import FingerprintConfig

from .base import FileHandler
from .image_handler import ImageFileHandler

logger = logging.getLogger(__name__)

# Sample one keyframe per this many seconds of video. A small fixed cadence
# keeps the keyframe grid stable across re-encodes/trims (see module docstring).
# Exposed via FingerprintConfig.video_keyframe_interval so a caller can tune it;
# the default mirrors the config default so no-arg construction is unchanged.
_DEFAULT_KEYFRAME_INTERVAL_SECONDS = 1.0
# Hard cap on decoded keyframes so an hours-long file cannot blow up memory or
# the signal length. 0 = unlimited. Mirrors FingerprintConfig.video_max_keyframes.
_DEFAULT_MAX_KEYFRAMES = 600

# Canonical per-frame grid, reused from the image handler so a video keyframe
# and a still image are canonicalized identically.
_CANONICAL_SIZE = ImageFileHandler.canonical_size  # (256, 256)


@dataclass(frozen=True)
class VideoPayload:
    """Decoded keyframe sequence as a stack of canonical grayscale grids.

    ``frames`` has shape ``(num_keyframes, height, width)`` in canonical
    (256x256) grayscale, float32 in the raw 0..255 intensity range (the
    normalisation to the FFT signal happens in :meth:`VideoFileHandler.to_signal`,
    matching how the image handler splits load/normalise).
    """

    frames: np.ndarray
    frame_count: int
    sampled_keyframes: int
    duration_seconds: float
    decoder: str
    keyframe_interval_seconds: float


class VideoFileHandler(FileHandler):
    name = "video"
    # Above image (60) / audio (70): a container that is genuinely a video must
    # win over a frame-level still-image guess on the same bytes. Unrelated files
    # still score 0 here (see ``can_handle``), so existing routing is unchanged.
    priority = 75
    canonical_size = _CANONICAL_SIZE
    supported_mime_prefixes = {"video/"}
    supported_mime_types = {
        "video/mp4",
        "video/quicktime",
        "video/x-matroska",
        "video/webm",
    }
    supported_extensions = {".mp4", ".mov", ".mkv", ".webm"}

    def __init__(
        self,
        keyframe_interval_seconds: float | None = None,
        max_keyframes: int | None = None,
    ) -> None:
        # ``None`` -> config defaults, so no-arg discovery construction matches
        # the documented defaults while a caller (or config wiring) can tune the
        # cadence and the keyframe cap.
        config = FingerprintConfig()
        if keyframe_interval_seconds is None:
            keyframe_interval_seconds = getattr(
                config, "video_keyframe_interval", _DEFAULT_KEYFRAME_INTERVAL_SECONDS
            )
        if max_keyframes is None:
            max_keyframes = getattr(config, "video_max_keyframes", _DEFAULT_MAX_KEYFRAMES)
        if keyframe_interval_seconds <= 0:
            raise ValueError("keyframe_interval_seconds must be positive")
        if max_keyframes < 0:
            raise ValueError("max_keyframes must be non-negative (0 = unlimited)")
        self.keyframe_interval_seconds = float(keyframe_interval_seconds)
        self.max_keyframes = int(max_keyframes)

    def configure(self, config: FingerprintConfig) -> None:
        # Honor config-derived tuning if/when it is added; ``getattr`` keeps this
        # forward-compatible without requiring the config fields to exist yet, so
        # the default path is unchanged.
        self.keyframe_interval_seconds = float(
            getattr(config, "video_keyframe_interval", self.keyframe_interval_seconds)
        )
        self.max_keyframes = int(getattr(config, "video_max_keyframes", self.max_keyframes))

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
        if sample and cls._sniff_video(sample):
            return 0.90
        return 0.0

    @staticmethod
    def _sniff_video(sample: bytes) -> bool:
        """Magic-byte sniff for the four supported containers.

        Returns ``False`` for anything else so unrelated files (and even other
        ISO-BMFF brands we do not claim) never route here.
        """

        # ISO base media (MP4/MOV): an ``ftyp`` box at offset 4. Match the brand
        # to our containers so a non-video ISO-BMFF (e.g. HEIF) does not route in.
        if len(sample) >= 12 and sample[4:8] == b"ftyp":
            brand = sample[8:12]
            if brand in {
                b"isom",
                b"iso2",
                b"mp41",
                b"mp42",
                b"avc1",
                b"qt  ",  # QuickTime / .mov
                b"M4V ",
            }:
                return True
        # Matroska / WebM share the EBML magic; the DocType disambiguates but the
        # magic alone is a strong-enough signal for routing here.
        if sample.startswith(b"\x1a\x45\xdf\xa3"):
            return True
        return False

    def load(self, path: str | Path) -> VideoPayload:
        """Decode keyframes at a fixed cadence into canonical grayscale grids.

        The video decoder (``av``, i.e. PyAV/ffmpeg) is imported LAZILY here so a
        core-only install can import this module and discover the handler without
        the dep. If it is absent, raise :class:`MissingDependencyError` for the
        ``video`` extra rather than degrade silently.
        """

        try:
            import av
        except ImportError as exc:
            logger.warning(
                "missing optional dependency %s (extra %s) for video fingerprinting",
                "av",
                "video",
            )
            raise MissingDependencyError(
                "PyAV (av) is required for video fingerprinting; install with "
                "'pip install \"fingerprint-engine[video]\"'",
                package="av",
                extra="video",
            ) from exc

        # Reuse the IMAGE handler's canonicalization so a keyframe is reduced
        # exactly like a still image (256x256 grayscale via PIL Lanczos). PIL is
        # the ``image`` extra; the lazy import here gives a clear message if a
        # video backend is present but PIL is not.
        try:
            from PIL import Image
        except ImportError as exc:
            raise MissingDependencyError(
                "Pillow is required to canonicalize video keyframes; install with "
                "'pip install \"fingerprint-engine[video]\"'",
                package="Pillow",
                extra="video",
            ) from exc

        resampling = getattr(Image, "Resampling", Image).LANCZOS
        target = self.canonical_size

        frames: list[np.ndarray] = []
        frame_count = 0
        duration_seconds = 0.0

        with av.open(str(path)) as container:
            stream = next((s for s in container.streams if s.type == "video"), None)
            if stream is None:
                raise MissingDependencyError(
                    "no decodable video stream found in the input container",
                    package="av",
                    extra="video",
                )
            # Frames-per-keyframe stride from the container frame rate; fall back
            # to sampling every frame if the rate is unknown.
            average_rate = float(stream.average_rate) if stream.average_rate else 0.0
            stride = max(1, int(round(average_rate * self.keyframe_interval_seconds))) if average_rate else 1

            for index, frame in enumerate(container.decode(stream)):
                frame_count = index + 1
                if index % stride != 0:
                    continue
                if self.max_keyframes and len(frames) >= self.max_keyframes:
                    break
                # Decoding a video stream yields VideoFrames; this narrows the
                # PyAV-17 stub union (VideoFrame|AudioFrame|SubtitleSet) so the
                # VideoFrame-only .to_image() type-checks, and defensively skips
                # any non-video frame instead of erroring.
                if not isinstance(frame, av.VideoFrame):
                    continue
                image = frame.to_image()  # PIL.Image in native size/mode
                grayscale = image.convert("L").resize(target, resampling)
                frames.append(np.asarray(grayscale, dtype=np.float32))

            if stream.duration is not None and stream.time_base is not None:
                duration_seconds = float(stream.duration * stream.time_base)

        if not frames:
            # A container that decoded zero frames is unusable for a keyframe
            # fingerprint; fail loud rather than emit an empty signal.
            raise MissingDependencyError(
                "video decoded zero frames; cannot build a keyframe fingerprint",
                package="av",
                extra="video",
            )

        stack = np.stack(frames, axis=0)
        height, width = target[1], target[0]
        return VideoPayload(
            frames=stack.reshape(len(frames), height, width),
            frame_count=frame_count,
            sampled_keyframes=len(frames),
            duration_seconds=duration_seconds,
            decoder="av",
            keyframe_interval_seconds=self.keyframe_interval_seconds,
        )

    def to_signal(self, payload: VideoPayload) -> np.ndarray:
        """Flatten the keyframe stack into one time-ordered 1D signal.

        Each canonical frame is normalised with the SAME transform the image
        handler applies to a still (``(pixels - 127.5) / 127.5``), then the
        frames are concatenated in time order. The result is a keyframe-sequence
        signal the shared FFT pipeline fingerprints unchanged.
        """

        if payload.frames.size == 0:
            return np.zeros(1, dtype=np.float32)
        normalised = (payload.frames.astype(np.float32) - 127.5) / 127.5
        return normalised.reshape(-1)

    def metadata(self, payload: VideoPayload) -> dict[str, object]:
        return {
            "frame_count": payload.frame_count,
            "sampled_keyframes": payload.sampled_keyframes,
            "duration_seconds": round(payload.duration_seconds, 6),
            "keyframe_interval_seconds": payload.keyframe_interval_seconds,
            "canonical_width": self.canonical_size[0],
            "canonical_height": self.canonical_size[1],
            "decoder": payload.decoder,
            "signal_strategy": "keyframe_sequence_canonical_256x256_grayscale",
        }
