"""Handler for audio files using decoded sample signals."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fingerprint_engine.core.exceptions import MissingDependencyError

from .base import FileHandler

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AudioPayload:
    samples: np.ndarray
    sample_rate: int
    channels: int
    decoder: str


class AudioFileHandler(FileHandler):
    name = "audio"
    priority = 70
    # Deliberately NARROW to the formats the loaders actually support (WAV via
    # scipy, MP3 via pydub/ffmpeg). The previous broad ``audio/`` MIME prefix
    # routed every audio container (.ogg/.flac/.m4a/.aac) here, where load()
    # then force-decoded them as MP3 and produced garbage or failed; those now
    # score 0.0 and fall through to text/binary deliberately. Exact MIME types
    # (not the prefix) keep the WAV/MP3 routing intact.
    supported_mime_types = {
        "audio/wav",
        "audio/x-wav",
        "audio/wave",
        "audio/vnd.wave",
        "audio/mpeg",
        "audio/mp3",
        "audio/x-mp3",
        "audio/mpeg3",
        "audio/x-mpeg-3",
    }
    supported_extensions = {".wav", ".wave", ".mp3"}

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
            sample.startswith(b"RIFF") and sample[8:12] == b"WAVE"
            or sample.startswith(b"ID3")
            or sample[:2] in {b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"}
        ):
            return 0.90
        return 0.0

    def load(self, path: str | Path) -> AudioPayload:
        suffix = Path(path).suffix.lower()
        if suffix in {".wav", ".wave"}:
            return self._load_wav(path)
        if suffix == ".mp3":
            return self._load_mp3(path)

        try:
            return self._load_wav(path)
        except Exception:
            return self._load_mp3(path)

    def to_signal(self, payload: AudioPayload) -> np.ndarray:
        return np.asarray(payload.samples, dtype=np.float32)

    def metadata(self, payload: AudioPayload) -> dict[str, object]:
        sample_count = int(payload.samples.size)
        duration = sample_count / float(payload.sample_rate) if payload.sample_rate else 0.0
        return {
            "sample_rate": payload.sample_rate,
            "channels": payload.channels,
            "duration_seconds": round(duration, 6),
            "decoder": payload.decoder,
            "signal_strategy": "decoded_mono_audio_samples",
        }

    @staticmethod
    def _load_wav(path: str | Path) -> AudioPayload:
        try:
            from scipy.io import wavfile
        except ImportError as exc:
            logger.warning(
                "missing optional dependency %s (extra %s) for WAV fingerprinting",
                "scipy",
                "audio",
            )
            raise MissingDependencyError(
                "scipy is required for WAV fingerprinting; install with "
                "'pip install fingerprint_engine[audio]'",
                package="scipy",
                extra="audio",
            ) from exc

        sample_rate, data = wavfile.read(path)
        array = np.asarray(data)
        channels = int(array.shape[1]) if array.ndim > 1 else 1
        if array.ndim > 1:
            array = array.astype(np.float32).mean(axis=1)
        else:
            array = array.astype(np.float32)

        if np.issubdtype(data.dtype, np.integer):
            max_value = float(np.iinfo(data.dtype).max)
            if max_value > 0:
                array = array / max_value
        else:
            max_abs = float(np.max(np.abs(array))) if array.size else 0.0
            if max_abs > 1.0:
                array = array / max_abs

        return AudioPayload(
            samples=np.nan_to_num(array.astype(np.float32), nan=0.0),
            sample_rate=int(sample_rate),
            channels=channels,
            decoder="scipy.io.wavfile",
        )

    @staticmethod
    def _load_mp3(path: str | Path) -> AudioPayload:
        try:
            from pydub import AudioSegment
        except ImportError as exc:
            logger.warning(
                "missing optional dependency %s (extra %s) for MP3 fingerprinting",
                "pydub",
                "audio",
            )
            raise MissingDependencyError(
                "pydub plus ffmpeg is required for MP3 fingerprinting; install with "
                "'pip install fingerprint_engine[audio]'",
                package="pydub",
                extra="audio",
            ) from exc

        # No hardcoded ``format=`` so ffmpeg sniffs the real container. The
        # previous force-decode-as-mp3 turned any non-MP3 input routed here into
        # garbage; letting ffmpeg detect the format keeps a genuinely-MP3 file
        # decoding correctly while a mislabeled one fails cleanly instead.
        segment = AudioSegment.from_file(path)
        samples = np.asarray(segment.get_array_of_samples(), dtype=np.float32)
        channels = int(segment.channels)
        if channels > 1:
            # Truncate to whole frames before de-interleaving: a corrupt/truncated
            # stream can yield a sample count not divisible by the channel count,
            # and a bare reshape((-1, channels)) would raise ValueError. Dropping
            # the trailing partial frame degrades to a best-effort mono signal
            # instead of failing the whole handler.
            usable = (samples.size // channels) * channels
            samples = samples[:usable].reshape((-1, channels)).mean(axis=1)
        max_value = float(1 << (8 * segment.sample_width - 1))
        if max_value > 0:
            samples = samples / max_value
        return AudioPayload(
            samples=np.nan_to_num(samples.astype(np.float32), nan=0.0),
            sample_rate=int(segment.frame_rate),
            channels=channels,
            decoder="pydub.ffmpeg",
        )
