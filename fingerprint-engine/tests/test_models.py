"""Tests for FingerprintConfig.validate covering every error branch."""

from __future__ import annotations

import dataclasses

import pytest

from fingerprint_engine.core.models import FingerprintConfig


def test_default_config_validates_cleanly() -> None:
    # The shipped defaults must pass validation without raising.
    FingerprintConfig().validate()


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"window_size": 7}, "window_size must be at least 8"),
        ({"hop_size": 0}, "hop_size must be at least 1"),
        ({"max_peaks_per_frame": 0}, "max_peaks_per_frame must be at least 1"),
        ({"constellation_fanout": 0}, "constellation_fanout must be at least 1"),
        ({"min_delta_t": -1}, "min_delta_t must be non-negative"),
        ({"min_delta_t": 10, "max_delta_t": 5}, "max_delta_t must be >= min_delta_t"),
        ({"hash_bits": 0}, "hash_bits must be between 1 and 64"),
        ({"hash_bits": 65}, "hash_bits must be between 1 and 64"),
        ({"max_signal_samples": 100}, "max_signal_samples must be >= window_size"),
        ({"min_time_frames": 0}, "min_time_frames must be at least 1"),
        ({"min_window_size": 7}, "min_window_size must be at least 8"),
        ({"min_window_size": 8192}, "min_window_size must be <= window_size"),
        ({"peak_percentile": -0.1}, "peak_percentile must be between 0.0 and 100.0"),
        ({"peak_percentile": 150.0}, "peak_percentile must be between 0.0 and 100.0"),
        ({"peak_threshold": -0.5}, "peak_threshold must be non-negative"),
    ],
)
def test_validate_rejects_out_of_range(overrides: dict[str, object], match: str) -> None:
    config = dataclasses.replace(FingerprintConfig(), **overrides)
    with pytest.raises(ValueError, match=match):
        config.validate()


@pytest.mark.parametrize("boundary", [0.0, 100.0])
def test_peak_percentile_boundaries_accepted(boundary: float) -> None:
    # The inclusive [0, 100] bounds must be valid, not just the interior.
    dataclasses.replace(FingerprintConfig(), peak_percentile=boundary).validate()


def test_peak_threshold_zero_accepted() -> None:
    # Zero is a valid (degenerate) multiplier; only negatives are rejected.
    dataclasses.replace(FingerprintConfig(), peak_threshold=0.0).validate()


def test_hash_changing_fields_match_effective_format_version() -> None:
    # HASH_CHANGING_FIELDS must stay in lock-step with effective_format_version:
    # every declared field changes the version when set non-default, and a few
    # representative inert fields do not. Guards a future hash-changing flag added
    # without an offset (which would silently change hashes at the default stamp).
    from fingerprint_engine.core.models import (
        FINGERPRINT_FORMAT_VERSION,
        HASH_CHANGING_FIELDS,
        effective_format_version,
    )

    base = FINGERPRINT_FORMAT_VERSION
    assert effective_format_version(FingerprintConfig()) == base

    changers = {"freq_quantization": 2, "window_bank": (512, 1024), "image_mode": "phash"}
    assert set(changers) == set(HASH_CHANGING_FIELDS)
    for field, value in changers.items():
        cfg = dataclasses.replace(FingerprintConfig(), **{field: value})
        assert effective_format_version(cfg) != base, field

    # Representative INERT fields (tuning + resource limits) must NOT move it.
    for field, value in {
        "window_size": 2048,
        "peak_threshold": 2.0,
        "max_file_size_bytes": 1024,
        "max_image_pixels": 1000,
    }.items():
        cfg = dataclasses.replace(FingerprintConfig(), **{field: value})
        assert effective_format_version(cfg) == base, field
