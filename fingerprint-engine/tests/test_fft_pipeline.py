from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fingerprint_engine.core.fft_pipeline import FFTFingerprintPipeline
from fingerprint_engine.core.models import FingerprintConfig


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
