"""Shared FFT-equivalent fingerprinting pipeline."""

from __future__ import annotations

import hashlib
import warnings
from collections.abc import Iterable
from dataclasses import replace

import numpy as np

from .models import ConstellationHash, FingerprintConfig, LandmarkPoint


class FFTFingerprintPipeline:
    """Transforms a 1D signal into landmark peaks and constellation hashes.

    ``fixed_window`` marks the configured ``window_size``/``hop_size`` as
    *authoritative* for a content type (see
    :meth:`Fingerprinter._build_handler_pipelines`): such a window must NOT be
    shrunk as a function of per-file length, otherwise two copies of the same
    content at different lengths land on different time grids and silently fail
    to match. The window is still adapted as a last resort when a signal is too
    short to yield even one usable frame at the declared window (so tiny inputs
    get *some* hashes rather than zero); that rare adaptation emits a one-line
    ``RuntimeWarning`` so it is never silent. The default (``fixed_window`` is
    False) keeps the original length-adaptive behaviour, which is what the
    global ``--window-size`` override and the audio 4096 path rely on.
    """

    def __init__(
        self,
        config: FingerprintConfig | None = None,
        *,
        fixed_window: bool = False,
    ) -> None:
        self.config = config or FingerprintConfig()
        self.config.validate()
        self.fixed_window = fixed_window

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
        else:
            # Zero-variance (constant) signal: featureless, regardless of the
            # constant's value. The old `array / max(abs)` branch normalised any
            # nonzero constant to an all-ones array, so two DIFFERENT constants
            # (e.g. full(N, 5.0) vs full(N, 99.0)) produced byte-identical
            # spectra and hashes -- distinct files would mutually false-match.
            # Return zeros so the spectrogram.max() <= 0 guard yields no peaks
            # (0 hashes) and the 0-hash RuntimeWarning in Fingerprinter fires.
            array = np.zeros_like(array)
        return array.astype(np.float32, copy=False)

    def _frames_at(self, signal_len: int, window: int, hop: int) -> float:
        """Number of sliding frames a signal of ``signal_len`` yields."""

        if signal_len <= window:
            return 1.0
        return (signal_len - window) / hop + 1

    def _effective_window_hop(self, signal_len: int, *, warn: bool = False) -> tuple[int, int]:
        """Resolve the (window, hop) actually used for ``signal_len`` samples.

        For an *authoritative* window (``fixed_window``; see the class
        docstring) the declared window/hop is returned UNCHANGED, so the same
        content type lands on the same time grid regardless of length and
        excerpts/truncations of the same content still match. The only
        exception is a signal too short to yield even one usable frame pair at
        the declared window: there we fall back to the adaptive shrink so tiny
        inputs still get *some* hashes rather than zero, and -- when ``warn`` is
        set -- emit a one-line ``RuntimeWarning`` so the rare adaptation is
        never silent. ``warn`` is set only on the real compute path
        (:meth:`spectrogram`); the metadata query :meth:`effective_params`
        leaves it off so the warning fires once per fingerprint, not twice.

        Otherwise (the default and the global ``--window-size`` override) the
        length-adaptive shrink is applied as before.
        """

        if not self.fixed_window:
            return self._adaptive_window_hop(signal_len)

        window = self.config.window_size
        hop = max(1, self.config.hop_size)
        # A usable fingerprint needs at least one anchor->target pair that spans
        # ``min_delta_t`` frames, i.e. min_delta_t + 1 frames (and >= 2 in any
        # case, since a single frame can carry no time structure). Above that
        # floor the declared window is authoritative and must not move.
        min_usable_frames = max(2, self.config.min_delta_t + 1)
        if signal_len <= 0 or self._frames_at(signal_len, window, hop) >= min_usable_frames:
            return window, hop

        adapted_window, adapted_hop = self._adaptive_window_hop(signal_len)
        if warn and (adapted_window, adapted_hop) != (window, hop):
            warnings.warn(
                f"signal too short ({signal_len} samples) for the fixed "
                f"window {window}/hop {hop}; adapting to window "
                f"{adapted_window}/hop {adapted_hop} (this file may not align "
                "with same-content files at the fixed window)",
                RuntimeWarning,
                stacklevel=2,
            )
        return adapted_window, adapted_hop

    def _adaptive_window_hop(self, signal_len: int) -> tuple[int, int]:
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

        current_frames = self._frames_at(signal_len, window, hop)
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
        When the multi-resolution window bank is active there is no single
        window, so the smallest bank window's effective (window, hop) is
        reported -- purely informational metadata, not load-bearing for matching.
        """

        if self.config.window_bank:
            return self._bank_pipeline(min(self.config.window_bank)).effective_params(signal)
        return self._effective_window_hop(self.normalize_signal(signal).size)

    def spectrogram(self, signal: np.ndarray | Iterable[float]) -> np.ndarray:
        """Apply sliding windows and rFFT to build a time x frequency matrix."""

        normalized = self.normalize_signal(signal)
        window_size, hop_size = self._effective_window_hop(normalized.size, warn=True)
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
        """Identify deterministic local maxima above an adaptive threshold.

        Vectorized equivalent of the original per-candidate scan: instead of
        recomputing a clipped 3x3 ``neighborhood.max()`` for every candidate bin
        in Python, the 3x3 clipped local maximum is computed once for the whole
        matrix as a separable pair of 1-D 3-window maxima (max along frequency,
        then along time). Each 1-D pass pads with ``-inf`` and takes the
        elementwise max of the three shifted slices; ``-inf`` padding never wins
        a ``max``, so each cell's window max is taken over exactly the same
        edge-CLIPPED 3x3 block the original used. A separable max equals the 2-D
        block max because ``max`` is associative/commutative, and float32
        ``max`` is pure selection (no arithmetic) so equality is exact -- the
        produced landmarks are byte-identical to the original implementation,
        without materializing a large strided neighborhood view.
        """

        matrix = np.asarray(spectrogram, dtype=np.float32)
        if matrix.ndim != 2 or matrix.size == 0 or float(matrix.max()) <= 0.0:
            return []

        mean = float(matrix.mean())
        std = float(matrix.std())
        percentile = float(np.percentile(matrix, self.config.peak_percentile))
        threshold = max(mean + self.config.peak_threshold * std, percentile)

        # 3x3 clipped local maximum for every cell, computed separably. Padding
        # with -inf keeps each windowed max equal to the original edge-clipped
        # ``matrix[t-1:t+2, f-1:f+2].max()`` (padding can never be the maximum),
        # so ``matrix == local_max`` reproduces ``magnitude >= neighborhood.max()``
        # exactly (a cell is in its own neighborhood, so the max is >= the cell).
        rows, cols = matrix.shape
        pad_freq = np.full((rows, cols + 2), -np.inf, dtype=np.float32)
        pad_freq[:, 1:-1] = matrix
        col_max = np.maximum(np.maximum(pad_freq[:, :-2], pad_freq[:, 1:-1]), pad_freq[:, 2:])
        pad_time = np.full((rows + 2, cols), -np.inf, dtype=np.float32)
        pad_time[1:-1, :] = col_max
        local_max = np.maximum(np.maximum(pad_time[:-2, :], pad_time[1:-1, :]), pad_time[2:, :])

        # A peak qualifies iff it is its own 3x3 max, strictly positive, and at
        # or above the threshold -- identical predicate to the original loop.
        qualifies = (matrix >= local_max) & (matrix > 0.0) & (matrix >= threshold)

        peaks: list[LandmarkPoint] = []
        max_per_frame = self.config.max_peaks_per_frame
        time_count = matrix.shape[0]
        for time_index in range(time_count):
            frame_bins = np.flatnonzero(qualifies[time_index])
            if frame_bins.size == 0:
                continue
            # ``round(float(value), 6)`` reproduces the original stored (and
            # sorted-on) magnitude exactly: float() on a float32 is lossless and
            # round() then matches the per-element Python round.
            row = matrix[time_index]
            frame_candidates = [
                LandmarkPoint(
                    time_index=time_index,
                    frequency_bin=int(frequency_bin),
                    magnitude=round(float(row[frequency_bin]), 6),
                )
                for frequency_bin in frame_bins
            ]
            frame_candidates.sort(key=lambda item: (-item.magnitude, item.frequency_bin))
            peaks.extend(frame_candidates[:max_per_frame])

        peaks.sort(key=lambda item: (item.time_index, item.frequency_bin, -item.magnitude))
        return peaks

    def build_hashes(
        self,
        peaks: list[LandmarkPoint],
        *,
        window_tag: int | None = None,
    ) -> list[ConstellationHash]:
        """Build constellation pairs and hash them into compact integer codes.

        ``window_tag`` is forwarded to :meth:`hash_pair`. ``None`` (the default)
        is the byte-identical single-window path; a window size folds that window
        into every code so it only collides with same-window codes (the
        multi-resolution window bank, see :meth:`fingerprint_signal`).
        """

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
                hash_code = self.hash_pair(
                    anchor.frequency_bin, target.frequency_bin, delta_t, window_tag=window_tag
                )
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

    def hash_pair(self, freq1: int, freq2: int, delta_t: int, *, window_tag: int | None = None) -> int:
        """Hash a peak-pair tuple into the configured number of bits.

        When ``config.freq_quantization`` is 1 (the default) the exact
        ``(freq1, freq2, delta_t)`` tuple is hashed -- byte-identical to the
        behaviour before the flag existed. When it is >1 the two frequency bins
        are snapped to a coarser grid (``bin // freq_quantization``) before
        packing, so peaks that drift by less than one coarse band (e.g. from a
        JPEG re-encode or a small text edit) hash to the same code and the true
        match survives the shift. ``delta_t`` is left untouched. The packed
        ``freq1``/``freq2`` therefore carry a *band index*, not a raw bin, when
        quantization is enabled.

        ``window_tag`` is the multi-resolution window-bank fold (see
        :meth:`fingerprint_signal` and :attr:`FingerprintConfig.window_bank`).
        When ``None`` (the default and the entire single-window path) the
        payload is exactly the three packed fields above -- byte-identical to the
        pre-bank hash. When a window size is supplied, it is appended to the
        payload before hashing, so a code derived at window ``w`` can only
        collide with another code derived at window ``w`` (codes from different
        bank windows live in disjoint regions of the hash space).
        """

        quant = self.config.freq_quantization
        if quant > 1:
            freq1 = int(freq1) // quant
            freq2 = int(freq2) // quant
        payload = (
            int(freq1).to_bytes(4, "big", signed=False)
            + int(freq2).to_bytes(4, "big", signed=False)
            + int(delta_t).to_bytes(4, "big", signed=False)
        )
        if window_tag is not None:
            payload += int(window_tag).to_bytes(4, "big", signed=False)
        digest = hashlib.blake2b(payload, digest_size=8).digest()
        value = int.from_bytes(digest, "big", signed=False)
        if self.config.hash_bits == 64:
            return value
        return value & ((1 << self.config.hash_bits) - 1)

    def _bank_hop(self, window: int) -> int:
        """Per-bank-window hop preserving the configured window:hop overlap ratio.

        The single-window overlap (e.g. 4096/1024 == 4) is applied to each bank
        window so all bank resolutions share the same fractional frame stride;
        clamped to >= 1.
        """

        ratio = self.config.window_size / max(1, self.config.hop_size)
        return max(1, round(window / ratio))

    def _bank_pipeline(self, window: int) -> FFTFingerprintPipeline:
        """A single-window sub-pipeline pinned to one bank window.

        Built with ``window_bank=None`` (so it runs the ordinary single-window
        path) and ``fixed_window=True`` (the bank window is authoritative: it is
        not shrunk as a function of per-file length, which is exactly what keeps
        the same content comparable across lengths at a given resolution). The
        hop preserves the configured overlap ratio.
        """

        sub_config = replace(
            self.config,
            window_size=window,
            hop_size=self._bank_hop(window),
            window_bank=None,
        )
        return FFTFingerprintPipeline(sub_config, fixed_window=True)

    def fingerprint_signal(
        self,
        signal: np.ndarray | Iterable[float],
    ) -> tuple[list[LandmarkPoint], list[ConstellationHash]]:
        """Convenience wrapper for the full signal-to-hash flow.

        With ``config.window_bank`` unset (the default) this is the single-window
        path -- byte-identical to before the bank existed. With a bank set, the
        signal is fingerprinted once per bank window through a per-window
        sub-pipeline, and each window's size is folded into its hashes (so
        window-w codes only collide with window-w codes). The per-window landmark
        lists and hash lists are concatenated; landmarks are returned in
        ascending bank-window order and the hashes are deterministically sorted,
        so the output is reproducible for a given signal and config. A bank of N
        windows produces roughly N times the postings of the single-window path.
        """

        if not self.config.window_bank:
            matrix = self.spectrogram(signal)
            peaks = self.extract_peaks(matrix)
            hashes = self.build_hashes(peaks)
            return peaks, hashes

        all_peaks: list[LandmarkPoint] = []
        all_hashes: list[ConstellationHash] = []
        for window in sorted(self.config.window_bank):
            sub = self._bank_pipeline(window)
            matrix = sub.spectrogram(signal)
            peaks = sub.extract_peaks(matrix)
            hashes = sub.build_hashes(peaks, window_tag=window)
            all_peaks.extend(peaks)
            all_hashes.extend(hashes)
        all_hashes.sort(key=lambda item: (item.time_offset, item.hash_code, item.freq1, item.freq2))
        return all_peaks, all_hashes
