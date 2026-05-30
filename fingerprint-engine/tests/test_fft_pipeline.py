from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fingerprint_engine.core.fft_pipeline import FFTFingerprintPipeline
from fingerprint_engine.core.models import FingerprintConfig, LandmarkPoint


def _reference_extract_peaks(
    pipeline: FFTFingerprintPipeline, spectrogram: np.ndarray
) -> list[LandmarkPoint]:
    """Verbatim copy of the ORIGINAL pre-vectorization extract_peaks algorithm.

    Kept here as an independent oracle so the vectorized production
    implementation can be asserted byte-identical to the scalar one. Do not
    "optimize" this -- it must stay an exact transcription of the old loop.
    """

    matrix = np.asarray(spectrogram, dtype=np.float32)
    if matrix.ndim != 2 or matrix.size == 0 or float(matrix.max()) <= 0.0:
        return []

    mean = float(matrix.mean())
    std = float(matrix.std())
    percentile = float(np.percentile(matrix, pipeline.config.peak_percentile))
    threshold = max(mean + pipeline.config.peak_threshold * std, percentile)

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
        peaks.extend(frame_candidates[: pipeline.config.max_peaks_per_frame])

    peaks.sort(key=lambda item: (item.time_index, item.frequency_bin, -item.magnitude))
    return peaks


def _assert_identical(actual: list[LandmarkPoint], expected: list[LandmarkPoint]) -> None:
    assert len(actual) == len(expected)
    for got, want in zip(actual, expected, strict=True):
        assert got.time_index == want.time_index
        assert got.frequency_bin == want.frequency_bin
        # magnitude is the stored round(.,6) value; require exact equality.
        assert got.magnitude == want.magnitude


def test_vectorized_extract_peaks_matches_oracle_on_random_matrices() -> None:
    # Equivalence gate (a): the vectorized extract_peaks must be byte-identical
    # to the original scalar oracle across many seeded random matrices spanning
    # awkward shapes, plateaus, all-zero, clamped-negative, and tie cases.
    config = FingerprintConfig(
        window_size=64,
        hop_size=16,
        peak_threshold=0.5,
        peak_percentile=80.0,
        max_peaks_per_frame=4,
        constellation_fanout=3,
        max_delta_t=12,
    )
    pipeline = FFTFingerprintPipeline(config)
    rng = np.random.default_rng(20240529)

    shapes = [
        (1, 1),
        (1, 7),
        (7, 1),
        (2, 2),
        (3, 3),
        (5, 9),
        (9, 5),
        (16, 33),
        (33, 16),
        (1, 64),
        (64, 1),
    ]
    for _ in range(50):
        shapes.append((int(rng.integers(1, 20)), int(rng.integers(1, 40))))

    matrices: list[np.ndarray] = []
    for shape in shapes:
        # Generic positive-ish spectrogram-like values.
        matrices.append(np.abs(rng.standard_normal(shape)) * 3.0)
        # Values with negatives (extract_peaks clamps via the > 0 predicate).
        matrices.append(rng.standard_normal(shape) * 2.0)
        # All-equal plateau (every cell is its own neighborhood max -> ties).
        matrices.append(np.full(shape, 1.5))
        # All zero (max <= 0 -> empty).
        matrices.append(np.zeros(shape))
        # All negative (clamped to no peaks via > 0).
        matrices.append(np.full(shape, -1.0))
        # Quantized values to force exact magnitude ties across bins/frames.
        matrices.append(np.round(rng.standard_normal(shape) * 2.0, 1))

    for raw in matrices:
        matrix = raw.astype(np.float32)
        _assert_identical(
            pipeline.extract_peaks(matrix),
            _reference_extract_peaks(pipeline, matrix),
        )

    assert len(matrices) > 50


def test_vectorized_extract_peaks_matches_oracle_on_handler_spectrograms() -> None:
    # Equivalence gate (b): on the REAL spectrograms produced by each handler's
    # signal, the vectorized peaks AND the full constellation hashes must match
    # the oracle path exactly.
    from fingerprint_engine.handlers.binary_handler import BinaryFileHandler
    from fingerprint_engine.handlers.text_handler import TextFileHandler

    signals: list[tuple[str, np.ndarray, FingerprintConfig]] = []

    # Text handler signal from real source-like text.
    text_payload = (
        "def vectorize(matrix):\n"
        "    # compute a 3x3 local max\n"
        "    return matrix.max()\n"
    ) * 40
    text_signal = TextFileHandler().to_signal(text_payload)
    signals.append(
        ("text", text_signal, FingerprintConfig(window_size=512, hop_size=128))
    )

    # Binary handler signal from raw bytes.
    binary_bytes = bytes((i * 37 + 11) % 256 for i in range(8000))
    binary_signal = BinaryFileHandler().to_signal(binary_bytes)
    signals.append(
        ("binary", binary_signal, FingerprintConfig(window_size=512, hop_size=128))
    )

    # Image-like handler signal: a 2D raster flattened to 1D (matches how the
    # image handler reduces a canonical-resized image to a signal); generated
    # without PIL so this case always runs.
    rng = np.random.default_rng(99)
    raster = rng.integers(0, 256, size=(64, 64)).astype(np.float32)
    image_like = ((raster.ravel() - 127.5) / 127.5).astype(np.float32)
    signals.append(("image_like", image_like, FingerprintConfig()))

    # Audio-like handler signal: a tone-plus-harmonics waveform like decoded PCM.
    t = np.linspace(0, 30 * np.pi, 16000, dtype=np.float32)
    audio_like = (np.sin(t) + 0.4 * np.sin(2.5 * t) + 0.2 * np.sin(5.1 * t)).astype(
        np.float32
    )
    signals.append(("audio_like", audio_like, FingerprintConfig()))

    for name, signal, config in signals:
        pipeline = FFTFingerprintPipeline(config)
        matrix = pipeline.spectrogram(signal)

        new_peaks = pipeline.extract_peaks(matrix)
        oracle_peaks = _reference_extract_peaks(pipeline, matrix)
        _assert_identical(new_peaks, oracle_peaks)
        assert new_peaks, f"{name} produced no peaks; case is not exercising the path"

        # Full hash chain must be byte-identical via the production path vs the
        # oracle peaks fed through the same hash builder.
        new_hashes = pipeline.build_hashes(new_peaks)
        oracle_hashes = pipeline.build_hashes(oracle_peaks)
        assert new_hashes == oracle_hashes

        # And fingerprint_signal (which calls the vectorized extract_peaks) must
        # match the oracle path end to end.
        fp_peaks, fp_hashes = pipeline.fingerprint_signal(signal)
        _assert_identical(fp_peaks, oracle_peaks)
        assert fp_hashes == oracle_hashes


def test_pipeline_is_deterministic_for_same_signal() -> None:
    config = FingerprintConfig(
        window_size=64,
        hop_size=16,
        peak_threshold=0.5,
        peak_percentile=80.0,
        max_peaks_per_frame=4,
        constellation_fanout=3,
        max_delta_t=12,
    )
    pipeline = FFTFingerprintPipeline(config)
    x = np.linspace(0, 8 * np.pi, 2048, dtype=np.float32)
    signal = np.sin(x) + 0.25 * np.sin(3 * x)

    peaks_a, hashes_a = pipeline.fingerprint_signal(signal)
    peaks_b, hashes_b = pipeline.fingerprint_signal(signal)

    assert peaks_a == peaks_b
    assert hashes_a == hashes_b
    assert len(peaks_a) > 0
    assert len(hashes_a) > 0


def test_zero_signal_produces_no_peaks_or_hashes() -> None:
    pipeline = FFTFingerprintPipeline(FingerprintConfig(window_size=32, hop_size=8))

    peaks, hashes = pipeline.fingerprint_signal(np.zeros(256, dtype=np.float32))

    assert peaks == []
    assert hashes == []


def test_hash_bits_are_respected() -> None:
    pipeline = FFTFingerprintPipeline(FingerprintConfig(hash_bits=24))

    value = pipeline.hash_pair(12, 34, 5)

    assert 0 <= value < (1 << 24)


def test_short_signal_still_produces_searchable_hashes() -> None:
    # Regression: a signal far shorter than the default 4096 window used to
    # collapse to ~1 time frame and emit 0 hashes, making the file unsearchable.
    # Adaptive windowing must now spread peaks across time and yield codes.
    pipeline = FFTFingerprintPipeline(FingerprintConfig())  # default window 4096
    rng = np.random.default_rng(7)
    x = np.linspace(0, 40 * np.pi, 2532, dtype=np.float32)
    signal = np.sin(x) + 0.3 * np.sin(2.7 * x)
    signal = signal + 0.05 * rng.standard_normal(2532).astype(np.float32)

    peaks, hashes = pipeline.fingerprint_signal(signal)
    peaks_again, hashes_again = pipeline.fingerprint_signal(signal)

    assert len(peaks) > 0
    assert len(hashes) > 0
    # Peaks must span more than one frame, otherwise no pair can satisfy min_delta_t.
    assert max(point.time_index for point in peaks) >= 1
    # Adaptive windowing must stay deterministic for short signals.
    assert peaks == peaks_again
    assert hashes == hashes_again


def test_adaptive_window_only_shrinks_short_signals() -> None:
    config = FingerprintConfig()  # window_size=4096, hop_size=1024 (4:1 overlap)
    pipeline = FFTFingerprintPipeline(config)

    # Long signal: configured window/hop are returned untouched (no regression).
    assert pipeline._effective_window_hop(2_000_000) == (4096, 1024)

    # Short signal: window shrinks within bounds and the overlap ratio is kept.
    window, hop = pipeline._effective_window_hop(2532)
    assert config.min_window_size <= window < config.window_size
    assert round(window / hop) == round(config.window_size / config.hop_size)
    # Under the default (power-of-two) floor the adapted window is a power of two.
    assert window & (window - 1) == 0


def test_min_window_size_must_not_exceed_window_size() -> None:
    with pytest.raises(ValueError):
        FingerprintConfig(window_size=32, hop_size=8, min_window_size=64).validate()


def test_distinct_nonzero_constants_do_not_collide() -> None:
    # Defect A regression: a constant (zero-variance) signal is featureless, but
    # the old `array / max(abs)` branch normalised any nonzero constant to an
    # all-ones array, so two DIFFERENT constants produced byte-identical spectra
    # and constellation hashes -- distinct files would mutually false-match.
    # A zero-variance signal must now produce 0 hashes regardless of its value.
    pipeline = FFTFingerprintPipeline(FingerprintConfig())  # default window 4096

    _, hashes_five = pipeline.fingerprint_signal(np.full(4096, 5.0, dtype=np.float32))
    _, hashes_ninetynine = pipeline.fingerprint_signal(np.full(4096, 99.0, dtype=np.float32))

    assert hashes_five == []
    assert hashes_ninetynine == []
    # And the normalised constant is genuinely featureless (all zeros), not the
    # old all-ones array that the two constants used to collapse onto.
    normalized = pipeline.normalize_signal(np.full(4096, 5.0, dtype=np.float32))
    assert float(np.max(np.abs(normalized))) == 0.0


def test_fixed_window_is_authoritative_across_lengths() -> None:
    # Defect B regression at the pipeline level: with fixed_window=True the
    # declared window/hop is authoritative and must NOT shrink with per-file
    # length, so same-content files of differing length share a time grid.
    config = FingerprintConfig(window_size=512, hop_size=128)
    fixed = FFTFingerprintPipeline(config, fixed_window=True)
    adaptive = FFTFingerprintPipeline(config, fixed_window=False)

    # Two lengths that the ADAPTIVE path would map to different windows (the bug)
    # but that both clear the fixed window's tiny floor (>= 2 frames at 512/128).
    assert adaptive._effective_window_hop(2332) != adaptive._effective_window_hop(1136)
    assert fixed._effective_window_hop(2332) == (512, 128)
    assert fixed._effective_window_hop(1136) == (512, 128)


def test_fixed_window_falls_back_and_warns_for_tiny_signal() -> None:
    # The fixed window still adapts as a last resort for inputs too short to
    # yield a usable frame pair, and that rare adaptation is NOT silent.
    config = FingerprintConfig(window_size=512, hop_size=128)
    fixed = FFTFingerprintPipeline(config, fixed_window=True)

    # 200 samples -> < 2 frames at 512/128 -> fall back to the adaptive shrink.
    window, hop = fixed._effective_window_hop(200, warn=False)
    assert window < 512

    with pytest.warns(RuntimeWarning, match="fixed window"):
        fixed.spectrogram(np.linspace(0.0, 1.0, 200, dtype=np.float32))
