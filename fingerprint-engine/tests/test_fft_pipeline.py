from __future__ import annotations

import hashlib
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


def _reference_hash_pair(freq1: int, freq2: int, delta_t: int, hash_bits: int = 64) -> int:
    """Verbatim transcription of the ORIGINAL (pre-quantization) hash_pair.

    Independent oracle: the default ``freq_quantization == 1`` path MUST stay
    byte-identical to this. Do not refactor -- it must remain an exact copy of
    the exact-tuple hashing that existed before the opt-in flag.
    """

    payload = (
        int(freq1).to_bytes(4, "big", signed=False)
        + int(freq2).to_bytes(4, "big", signed=False)
        + int(delta_t).to_bytes(4, "big", signed=False)
    )
    value = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big", signed=False)
    if hash_bits == 64:
        return value
    return value & ((1 << hash_bits) - 1)


def _reference_hash_pair_with_window(
    freq1: int, freq2: int, delta_t: int, window: int, hash_bits: int = 64
) -> int:
    """Reference for a window-bank tagged hash: the exact tuple + window suffix.

    Mirrors the documented fold in ``hash_pair`` (the four packed big-endian
    uint32 fields). A tagged code must equal this so window-w codes only ever
    collide with window-w codes.
    """

    payload = (
        int(freq1).to_bytes(4, "big", signed=False)
        + int(freq2).to_bytes(4, "big", signed=False)
        + int(delta_t).to_bytes(4, "big", signed=False)
        + int(window).to_bytes(4, "big", signed=False)
    )
    value = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big", signed=False)
    if hash_bits == 64:
        return value
    return value & ((1 << hash_bits) - 1)


def test_freq_quantization_defaults_to_one_and_validates() -> None:
    # The flag defaults to 1 (off) and rejects values below 1.
    assert FingerprintConfig().freq_quantization == 1
    FingerprintConfig(freq_quantization=1).validate()
    FingerprintConfig(freq_quantization=8).validate()
    with pytest.raises(ValueError, match="freq_quantization"):
        FingerprintConfig(freq_quantization=0).validate()
    with pytest.raises(ValueError, match="freq_quantization"):
        FingerprintConfig(freq_quantization=-3).validate()


def test_default_freq_quantization_hashes_are_byte_identical() -> None:
    # DEFAULT-PRESERVING gate: with freq_quantization == 1 (the default) every
    # hash_pair output must equal the pre-flag exact-tuple reference, bit for
    # bit, across raw 64-bit and a narrowed hash_bits width.
    pipeline = FFTFingerprintPipeline(FingerprintConfig())  # default q == 1
    narrow = FFTFingerprintPipeline(FingerprintConfig(hash_bits=24))
    for freq1 in range(0, 400, 13):
        for freq2 in range(0, 400, 17):
            for delta_t in (1, 5, 17, 48):
                assert pipeline.hash_pair(freq1, freq2, delta_t) == _reference_hash_pair(
                    freq1, freq2, delta_t
                )
                assert narrow.hash_pair(freq1, freq2, delta_t) == _reference_hash_pair(
                    freq1, freq2, delta_t, hash_bits=24
                )


def test_default_freq_quantization_full_signal_hashes_unchanged() -> None:
    # End-to-end default-preserving proof: the constellation hash codes produced
    # by the full signal->hash pipeline at the default q == 1 must equal the
    # pre-flag reference applied to each pair's (freq1, freq2, delta_t).
    config = FingerprintConfig(window_size=512, hop_size=128)  # q defaults to 1
    pipeline = FFTFingerprintPipeline(config)
    rng = np.random.default_rng(7)
    x = np.linspace(0, 40 * np.pi, 16000, dtype=np.float32)
    signal = (np.sin(x) + 0.3 * np.sin(2.7 * x) + 0.05 * rng.standard_normal(16000)).astype(
        np.float32
    )

    _, hashes = pipeline.fingerprint_signal(signal)
    assert hashes, "case must produce hashes to be a meaningful gate"
    for item in hashes:
        assert item.hash_code == _reference_hash_pair(item.freq1, item.freq2, item.delta_t)


def test_freq_quantization_snaps_bins_to_a_coarser_grid() -> None:
    # With q > 1, hash_pair packs the BAND index (bin // q), so any two bins in
    # the same band collide while bins in different bands stay distinct.
    pipeline = FFTFingerprintPipeline(FingerprintConfig(freq_quantization=4))
    # 100..103 share band 25; 104 is band 26.
    base = pipeline.hash_pair(100, 200, 5)
    assert pipeline.hash_pair(101, 201, 5) == base  # both bins snap into the same bands
    assert pipeline.hash_pair(103, 203, 5) == base
    assert pipeline.hash_pair(104, 200, 5) != base  # freq1 crosses into the next band
    # The band-index value matches an exact hash of the snapped bins.
    assert base == _reference_hash_pair(100 // 4, 200 // 4, 5)


def test_freq_quantization_changes_hashes_but_still_self_matches() -> None:
    # q > 1 changes the produced hash codes relative to the default, yet remains
    # deterministic, so a file still matches itself.
    config = FingerprintConfig(window_size=512, hop_size=128)
    rng = np.random.default_rng(11)
    x = np.linspace(0, 50 * np.pi, 16000, dtype=np.float32)
    signal = (np.sin(x) + 0.4 * np.sin(3.1 * x) + 0.05 * rng.standard_normal(16000)).astype(
        np.float32
    )

    exact = FFTFingerprintPipeline(config)  # q == 1
    quantized = FFTFingerprintPipeline(FingerprintConfig(window_size=512, hop_size=128, freq_quantization=2))

    _, exact_hashes = exact.fingerprint_signal(signal)
    _, quant_hashes = quantized.fingerprint_signal(signal)
    _, quant_hashes_again = quantized.fingerprint_signal(signal)

    exact_codes = [item.hash_code for item in exact_hashes]
    quant_codes = [item.hash_code for item in quant_hashes]
    assert exact_codes  # the default path produced hashes
    assert quant_codes != exact_codes  # quantization actually changed the codes
    # Self-match: re-fingerprinting the same signal at q=2 is identical.
    assert quant_codes == [item.hash_code for item in quant_hashes_again]


# ---------------------------------------------------------------------------
# OPT-IN multi-resolution window bank.
# ---------------------------------------------------------------------------


def test_window_bank_defaults_to_none_and_validates() -> None:
    # The flag defaults to None (off) and validates the bank shape when set.
    assert FingerprintConfig().window_bank is None
    FingerprintConfig().validate()
    FingerprintConfig(window_bank=(512, 1024, 2048, 4096)).validate()
    FingerprintConfig(window_bank=(64,)).validate()  # min_window_size default is 16
    with pytest.raises(ValueError, match="non-empty"):
        FingerprintConfig(window_bank=()).validate()
    with pytest.raises(ValueError, match="distinct"):
        FingerprintConfig(window_bank=(512, 512)).validate()
    with pytest.raises(ValueError, match="below min_window_size"):
        FingerprintConfig(window_bank=(8,), min_window_size=16).validate()
    with pytest.raises(ValueError, match="max_window_bank_size"):
        FingerprintConfig(window_bank=(64, 128, 256, 512, 1024, 2048, 4096)).validate()
    with pytest.raises(ValueError, match="max_signal_samples"):
        FingerprintConfig(window_bank=(8192,), max_signal_samples=4096).validate()


def test_window_tag_is_byte_identical_when_none() -> None:
    # DEFAULT-PRESERVING gate at the hash level: window_tag=None (the single-
    # window path) must equal the pre-bank exact-tuple reference, and equal the
    # no-keyword call, bit for bit.
    pipeline = FFTFingerprintPipeline(FingerprintConfig())
    for freq1 in range(0, 300, 11):
        for freq2 in range(0, 300, 19):
            for delta_t in (1, 7, 48):
                base = pipeline.hash_pair(freq1, freq2, delta_t)
                assert base == pipeline.hash_pair(freq1, freq2, delta_t, window_tag=None)
                assert base == _reference_hash_pair(freq1, freq2, delta_t)


def test_window_tag_isolates_codes_per_window() -> None:
    # A tagged code differs from the untagged one and from a code tagged with a
    # different window, so window-w hashes only ever collide with window-w
    # hashes; the same tag is deterministic.
    pipeline = FFTFingerprintPipeline(FingerprintConfig())
    untagged = pipeline.hash_pair(10, 20, 5)
    tag_512 = pipeline.hash_pair(10, 20, 5, window_tag=512)
    tag_1024 = pipeline.hash_pair(10, 20, 5, window_tag=1024)
    assert tag_512 != untagged
    assert tag_512 != tag_1024
    assert tag_512 == pipeline.hash_pair(10, 20, 5, window_tag=512)
    # The tagged code is an exact hash of the packed payload + window suffix.
    assert tag_512 == _reference_hash_pair_with_window(10, 20, 5, 512)


def test_window_bank_off_is_byte_identical_to_single_window() -> None:
    # End-to-end DEFAULT-PRESERVING proof: with window_bank=None the full
    # signal->hash pipeline is identical to the path before the bank existed.
    config = FingerprintConfig(window_size=512, hop_size=128)  # window_bank defaults to None
    pipeline = FFTFingerprintPipeline(config)
    rng = np.random.default_rng(7)
    x = np.linspace(0, 40 * np.pi, 16000, dtype=np.float32)
    signal = (np.sin(x) + 0.3 * np.sin(2.7 * x) + 0.05 * rng.standard_normal(16000)).astype(
        np.float32
    )

    peaks, hashes = pipeline.fingerprint_signal(signal)
    assert hashes, "case must produce hashes to be a meaningful gate"
    # Every code equals the pre-bank, single-window exact reference (no window
    # folded in), and equals the explicit single-window build_hashes path.
    for item in hashes:
        assert item.hash_code == _reference_hash_pair(item.freq1, item.freq2, item.delta_t)
    matrix = pipeline.spectrogram(signal)
    direct = pipeline.build_hashes(pipeline.extract_peaks(matrix))
    assert hashes == direct


def test_window_bank_produces_matchable_per_window_hashes() -> None:
    # With a bank set, the pipeline emits hashes for EACH bank window, each code
    # folded with its window so a query at the same bank re-collides per window.
    bank = (256, 512, 1024)
    config = FingerprintConfig(window_size=512, hop_size=128, window_bank=bank)
    pipeline = FFTFingerprintPipeline(config)
    rng = np.random.default_rng(13)
    x = np.linspace(0, 50 * np.pi, 16000, dtype=np.float32)
    signal = (np.sin(x) + 0.4 * np.sin(3.1 * x) + 0.05 * rng.standard_normal(16000)).astype(
        np.float32
    )

    _, bank_hashes = pipeline.fingerprint_signal(signal)
    _, bank_hashes_again = pipeline.fingerprint_signal(signal)
    bank_codes = {item.hash_code for item in bank_hashes}
    assert bank_hashes, "bank must produce hashes"
    # Deterministic: re-fingerprinting the same signal at the same bank matches.
    assert [h.hash_code for h in bank_hashes] == [h.hash_code for h in bank_hashes_again]

    # The bank yields strictly more hashes than any single window alone (it is
    # the union over the bank), and the per-window codes are disjoint from the
    # single-window (untagged) codes of any one window.
    single = FFTFingerprintPipeline(FingerprintConfig(window_size=512, hop_size=128))
    _, single_hashes = single.fingerprint_signal(signal)
    single_codes = {item.hash_code for item in single_hashes}
    assert len(bank_hashes) > len(single_hashes)
    assert bank_codes.isdisjoint(single_codes)  # untagged vs window-tagged never collide

    # A query fingerprinted at the SAME bank shares codes back (self-match works
    # through the window fold); a query at a DIFFERENT bank shares none.
    same_bank = FFTFingerprintPipeline(config)
    _, same_hashes = same_bank.fingerprint_signal(signal)
    assert {h.hash_code for h in same_hashes} & bank_codes
    other = FFTFingerprintPipeline(
        FingerprintConfig(window_size=512, hop_size=128, window_bank=(2048,))
    )
    _, other_hashes = other.fingerprint_signal(signal)
    assert not ({h.hash_code for h in other_hashes} & bank_codes)


def test_window_bank_effective_params_reports_smallest_window() -> None:
    # effective_params (recorded in fingerprint metadata) reports the smallest
    # bank window's resolution when the bank is active -- informational only.
    config = FingerprintConfig(window_bank=(512, 1024, 2048))
    pipeline = FFTFingerprintPipeline(config)
    signal = np.linspace(0.0, 1.0, 16000, dtype=np.float32)
    window, _hop = pipeline.effective_params(signal)
    assert window == 512
