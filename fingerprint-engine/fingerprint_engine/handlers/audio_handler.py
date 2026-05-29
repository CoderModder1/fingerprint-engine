"""Handler for audio files using decoded sample signals."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .base import FileHandler


@dataclass(frozen=True)
class AudioPayload:
    samples: np.ndarray
    sample_rate: int
    channels: int
    decoder: str


class AudioFileHandler(FileHandler):
    name = "audio"
    priority = 70
    supported_mime_prefixes = {"audio/"}
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
            raise RuntimeError("scipy is required for WAV fingerprinting") from exc

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
            raise RuntimeError(
                "pydub plus ffmpeg is required for MP3 fingerprinting"
            ) from exc

        segment = AudioSegment.from_file(path, format="mp3")
        samples = np.asarray(segment.get_array_of_samples(), dtype=np.float32)
        channels = int(segment.channels)
        if channels > 1:
            samples = samples.reshape((-1, channels)).mean(axis=1)
        max_value = float(1 << (8 * segment.sample_width - 1))
        if max_value > 0:
            samples = samples / max_value
        return AudioPayload(
            samples=np.nan_to_num(samples.astype(np.float32), nan=0.0),
            sample_rate=int(segment.frame_rate),
            channels=channels,
            decoder="pydub.ffmpeg",
        )
