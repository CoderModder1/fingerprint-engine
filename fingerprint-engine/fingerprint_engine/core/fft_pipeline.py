"""Shared FFT-equivalent fingerprinting pipeline."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable

import numpy as np

from .models import ConstellationHash, FingerprintConfig, LandmarkPoint


class FFTFingerprintPipeline:
    """Transforms a 1D signal into landmark peaks and constellation hashes."""

    def __init__(self, config: FingerprintConfig | None = None) -> None:
        self.config = config or FingerprintConfig()
        self.config.validate()

    def bytes_to_signal(self, data: bytes) -> np.ndarray:
        """Default byte strategy used by binary-like handlers."""

        if not data:
            return np.zeros(self.config.window_size, dtype=np.float32)
        raw = np.frombuffer(data, dtype=np.uint8).astype(np.float32)
        return (raw - 127.5) / 127.5

    def normalize_signal(self, signal: np.ndarray | Iterable[float]) -> np.ndarray:
        """Return a finite, centered float32 signal with deterministic limiting."""

        array = np.asarray(list(signal) if not isinstance(signal, np.ndarray) else signal)
        array = np.ravel(array).astype(np.float32, copy=False)
        if array.size == 0:
            array = np.zeros(self.config.window_size, dtype=np.float32)
        array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)

        if array.size > self.config.max_signal_samples:
            indices = np.linspace(
                0,
                array.size - 1,
                self.config.max_signal_samples,
                dtype=np.int64,
            )
            array = array[indices]

        mean = float(array.mean())
        std = float(array.std())
        if std > 1e-8:
            array = (array - mean) / std
        elif np.max(np.abs(array)) > 0:
            array = array / float(np.max(np.abs(array)))
        return array.astype(np.float32, copy=False)

    def _effective_window_hop(self, signal_len: int) -> tuple[int, int]:
        """Adapt window/hop so short signals still yield enough time frames.

        A signal shorter than the configured window collapses to one or two
        frames, so no constellation pair can span ``min_delta_t`` and the
        fingerprint comes out empty (and therefore unsearchable). When the
        configured window would yield fewer than ``min_time_frames`` frames we
        shrink the window -- preserving the configured window:hop overlap ratio
        -- toward that target, clamped to ``[min_window_size, window_size]``.

        The shrunk window is rounded down to a power of two *when possible*; the
        ``min_window_size`` floor takes precedence, so a non-power-of-two floor
        yields a non-power-of-two window. That is purely a performance
        preference -- ``np.fft.rfft`` accepts any length, so it is never a
        correctness issue.

        Output depends only on ``signal_len`` and the (frozen) config, so the
        same input always maps to the same window/hop. Long signals that
        already reach ``min_time_frames`` are returned untouched, so
        normal-length inputs keep identical fingerprints. For very short signals
        (roughly ``min_window_size * (min_time_frames + ratio - 1) / ratio``
        samples or fewer) even the minimum window cannot reach
        ``min_time_frames``; the floor is used as a best effort and the 0-hash
        warning in :class:`Fingerprinter` surfaces any truly unsearchable file.
        """

        window = self.config.window_size
        hop = max(1, self.config.hop_size)
        min_frames = self.config.min_time_frames
        floor = self.config.min_window_size  # validate() guarantees floor <= window

        if signal_len <= 0 or min_frames <= 1:
            return window, hop

        current_frames = (signal_len - window) / hop + 1 if signal_len > window else 1.0
        if current_frames >= min_frames:
            return window, hop

        ratio = window / hop  # configured overlap, e.g. 4096/1024 == 4
        # Solve frames(w) = ratio*signal_len/w - ratio + 1 >= min_frames for w.
        target = ratio * signal_len / (min_frames + ratio - 1)
        new_window = max(floor, int(target))
        new_window = 1 << (new_window.bit_length() - 1)  # round down to power of two
        new_window = max(floor, min(window, new_window))  # floor wins if it is larger
        new_hop = max(1, round(new_window / ratio))
        return new_window, new_hop

    def effective_params(self, signal: np.ndarray | Iterable[float]) -> tuple[int, int]:
        """Public: the (window, hop) actually used to fingerprint ``signal``.

        Mirrors what :meth:`spectrogram` computes internally; recorded in
        fingerprint metadata so adaptive windowing is transparent and auditable.
        """

        return self._effective_window_hop(self.normalize_signal(signal).size)

    def spectrogram(self, signal: np.ndarray | Iterable[float]) -> np.ndarray:
        """Apply sliding windows and rFFT to build a time x frequency matrix."""

        normalized = self.normalize_signal(signal)
        window_size, hop_size = self._effective_window_hop(normalized.size)
        frame_count = max(1, int(np.ceil(max(1, normalized.size - window_size) / hop_size)) + 1)
        taper = np.hanning(window_size).astype(np.float32)
        spectra: list[np.ndarray] = []

        for frame_index in range(frame_count):
            start = frame_index * hop_size
            stop = start + window_size
            frame = np.zeros(window_size, dtype=np.float32)
            chunk = normalized[start:stop]
            if chunk.size:
                frame[: chunk.size] = chunk
            windowed = frame * taper
            magnitude = np.abs(np.fft.rfft(windowed))
            spectra.append(np.log1p(magnitude).astype(np.float32))

        matrix = np.vstack(spectra)
        return np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)

    def extract_peaks(self, spectrogram: np.ndarray) -> list[LandmarkPoint]:
        """Identify deterministic local maxima above an adaptive threshold."""

        matrix = np.asarray(spectrogram, dtype=np.float32)
        if matrix.ndim != 2 or matrix.size == 0 or float(matrix.max()) <= 0.0:
            return []

        mean = float(matrix.mean())
        std = float(matrix.std())
        percentile = float(np.percentile(matrix, self.config.peak_percentile))
        threshold = max(mean + self.config.peak_threshold * std, percentile)

        peaks: list[LandmarkPoint] = []
        time_count, freq_count = matrix.shape
        for time_index in range(time_count):
            frame_candidates: list[LandmarkPoint] = []
            row = matrix[time_index]
            candidate_bins = np.flatnonzero(row >= threshold)
            for frequency_bin in candidate_bins:
                magnitude = float(row[frequency_bin])
                t0 = max(0, time_index - 1)
                t1 = min(time_count, time_index + 2)
                f0 = max(0, int(frequency_bin) - 1)
                f1 = min(freq_count, int(frequency_bin) + 2)
                neighborhood = matrix[t0:t1, f0:f1]
                if magnitude >= float(neighborhood.max()) and magnitude > 0.0:
                    frame_candidates.append(
                        LandmarkPoint(
                            time_index=time_index,
                            frequency_bin=int(frequency_bin),
                            magnitude=round(magnitude, 6),
                        )
                    )

            frame_candidates.sort(key=lambda item: (-item.magnitude, item.frequency_bin))
            peaks.extend(frame_candidates[: self.config.max_peaks_per_frame])

        peaks.sort(key=lambda item: (item.time_index, item.frequency_bin, -item.magnitude))
        return peaks

    def build_hashes(self, peaks: list[LandmarkPoint]) -> list[ConstellationHash]:
        """Build constellation pairs and hash them into compact integer codes."""

        ordered = sorted(peaks, key=lambda item: (item.time_index, item.frequency_bin))
        hashes: list[ConstellationHash] = []
        for index, anchor in enumerate(ordered):
            targets: list[LandmarkPoint] = []
            for target in ordered[index + 1 :]:
                delta_t = target.time_index - anchor.time_index
                if delta_t < self.config.min_delta_t:
                    continue
                if delta_t > self.config.max_delta_t:
                    break
                targets.append(target)

            targets.sort(
                key=lambda item: (
                    item.time_index - anchor.time_index,
                    abs(item.frequency_bin - anchor.frequency_bin),
                    item.frequency_bin,
                )
            )
            for target in targets[: self.config.constellation_fanout]:
                delta_t = target.time_index - anchor.time_index
                hash_code = self.hash_pair(anchor.frequency_bin, target.frequency_bin, delta_t)
                hashes.append(
                    ConstellationHash(
                        hash_code=hash_code,
                        time_offset=anchor.time_index,
                        anchor_time=anchor.time_index,
                        target_time=target.time_index,
                        freq1=anchor.frequency_bin,
                        freq2=target.frequency_bin,
                        delta_t=delta_t,
                    )
                )

        hashes.sort(key=lambda item: (item.time_offset, item.hash_code, item.freq1, item.freq2))
        return hashes

    def hash_pair(self, freq1: int, freq2: int, delta_t: int) -> int:
        """Hash a peak-pair tuple into the configured number of bits."""

        payload = (
            int(freq1).to_bytes(4, "big", signed=False)
            + int(freq2).to_bytes(4, "big", signed=False)
            + int(delta_t).to_bytes(4, "big", signed=False)
        )
        digest = hashlib.blake2b(payload, digest_size=8).digest()
        value = int.from_bytes(digest, "big", signed=False)
        if self.config.hash_bits == 64:
            return value
        return value & ((1 << self.config.hash_bits) - 1)

    def fingerprint_signal(
        self,
        signal: np.ndarray | Iterable[float],
    ) -> tuple[list[LandmarkPoint], list[ConstellationHash]]:
        """Convenience wrapper for the full signal-to-hash flow."""

        matrix = self.spectrogram(signal)
        peaks = self.extract_peaks(matrix)
        hashes = self.build_hashes(peaks)
        return peaks, hashes
