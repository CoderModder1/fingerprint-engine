"""Handler for audio files using decoded sample signals."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import numpy as np

from .base import FileHandler, require_optional


@dataclass(frozen=True)
class AudioPayload:
    samples: np.ndarray
    sample_rate: int
    channels: int
    decoder: str


class AudioFileHandler(FileHandler):
    name = "audio"
    priority = 70
    # Multi-resolution window bank, ON BY DEFAULT for audio (v2 format). An
    # excerpt/clip re-normalises the signal and shifts the global time grid, so
    # its single-window hashes never collide with the whole file's -- audio
    # excerpt/clip recall at one fixed window is ~0. Fingerprinting at several
    # resolutions (the smallest window lets an excerpt align to its parent) lifts
    # excerpt recall to ~1.0 (independently verified), at ~4x the postings; the
    # 4096 entry preserves whole-file matching. A global FingerprintConfig.
    # window_bank overrides this, and an explicit --window-size disables it.
    default_window_bank = (512, 1024, 2048, 4096)
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

    def load(self, path: str | Path, *, content: bytes | None = None) -> AudioPayload:
        # Decode from the already-read bytes when provided (single-read path); the
        # decoders parse identical bytes from a BytesIO. suffix still comes from
        # the path so wav/mp3 routing is unchanged. None -> read the path.
        raw = self.read_content(path, content)
        suffix = Path(path).suffix.lower()
        if suffix in {".wav", ".wave"}:
            return self._load_wav(raw)
        if suffix == ".mp3":
            return self._load_mp3(raw)

        try:
            return self._load_wav(raw)
        except Exception:
            return self._load_mp3(raw)

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
    def _load_wav(raw: bytes) -> AudioPayload:
        wavfile = require_optional(
            "scipy.io.wavfile",
            package="scipy",
            extra="audio",
            message=(
                "scipy is required for WAV fingerprinting; install with "
                "'pip install fingerprint_engine[audio]'"
            ),
        )
        sample_rate, data = wavfile.read(BytesIO(raw))
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
    def _load_mp3(raw: bytes) -> AudioPayload:
        AudioSegment = require_optional(
            "pydub",
            package="pydub",
            extra="audio",
            message=(
                "pydub plus ffmpeg is required for MP3 fingerprinting; install with "
                "'pip install fingerprint_engine[audio]'"
            ),
        ).AudioSegment

        # No hardcoded ``format=`` so ffmpeg sniffs the real container. The
        # previous force-decode-as-mp3 turned any non-MP3 input routed here into
        # garbage; letting ffmpeg detect the format keeps a genuinely-MP3 file
        # decoding correctly while a mislabeled one fails cleanly instead.
        # from_file accepts a file-like; pydub spools it to a temp for ffmpeg, so
        # the same bytes are decoded whether they came from disk or the buffer.
        segment = AudioSegment.from_file(BytesIO(raw))
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
