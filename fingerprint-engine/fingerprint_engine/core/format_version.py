"""The hash-derivation format version: the single owner of the model.

This module concentrates the *hash-derivation* version (distinct from the
snapshot ``schema_version``, which versions the JSON container): the baseline
constant, the per-flag offsets, the effective-version computation, the explicit
set of hash-changing config fields, and the tolerant "absent ⇒ default" coercion
shared by every reader of a stamped/persisted version. Keeping it in one module
(rather than spread across ``models`` and the index backends) gives the version
model a single, greppable home. Re-exported from :mod:`fingerprint_engine.core.models`
so existing ``from ...core.models import FINGERPRINT_FORMAT_VERSION`` /
``effective_format_version`` imports are unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only; avoids a models<->this cycle
    from .models import FingerprintConfig

# Version of the HASH-DERIVATION format -- the rule set that turns a file's
# bytes into ``hash_code`` values. This is DISTINCT from the snapshot
# ``schema_version`` (which versions the *container* that serializes postings,
# see ``core/index/``): two builds can share a snapshot schema yet derive
# incompatible hash codes.
#
# Matching is only valid between a query and an index built with the SAME
# format version: a hash code carries no meaning across formats, so a
# cross-format "match" is a false result, not a weak one. Flipping any
# HASH-CHANGING default (the constellation packing, the per-handler windows, a
# new canonical image transform, ...) MUST bump this constant AND require
# re-indexing existing corpora -- see VERSIONING.md.
#
# The opt-in, default-off hash-changing flags do not change this baseline (their
# default values leave the derivation byte-identical); instead, when one is
# *enabled*, ``effective_format_version`` reports a DIFFERENT version for that
# config, so an index built with such a flag is detectably incompatible with a
# default index without flipping any default. The offsets below are the additive
# per-flag bumps; they are deliberately distinct so a config that enables several
# flags lands on a value distinct from enabling any one alone.
#
# Version 2 (2026-05-31): the default DERIVATION changed in two ways, so a v1
# corpus must be re-indexed (see VERSIONING.md):
#   1. signal/spectrogram reductions (mean/std/percentile) now accumulate in
#      float64 for cross-platform reproducibility (float32 reduction drift could
#      flip a borderline peak across numpy/BLAS/CPU). Output-identical for
#      non-audio handlers on real inputs; it shifts a near-zero-mean signal's
#      normalisation (e.g. audio), which is why audio hash codes change.
#   2. the audio handler now fingerprints with a multi-resolution window bank by
#      default (see AudioFileHandler.default_window_bank), so audio excerpt/clip
#      matching works out of the box. Non-audio handlers are output-identical to
#      v1; only the version STAMP advances (one version per index, so the bump is
#      global).
FINGERPRINT_FORMAT_VERSION = 2

# Key under which the effective format version is recorded in
# ``Fingerprint.config`` (a metadata-only stamp; it is NOT a tuning parameter,
# is NOT consumed by the FFT pipeline, and never enters a hash payload).
FORMAT_VERSION_KEY = "fingerprint_format_version"

# Additive offsets applied to ``FINGERPRINT_FORMAT_VERSION`` when an opt-in
# hash-changing flag is enabled (see the module note above).
_FORMAT_BUMP_FREQ_QUANTIZATION = 1000
_FORMAT_BUMP_WINDOW_BANK = 2000
_FORMAT_BUMP_IMAGE_PHASH = 4000

# The FingerprintConfig fields whose NON-default value changes the hash
# derivation (and therefore effective_format_version). The EXPLICIT single source
# of truth for "which config knobs are hash-changing", co-located with the
# version logic so the coupling is greppable rather than buried in the body of
# effective_format_version below. Adding a new hash-changing flag means adding it
# here AND giving it an offset above; test_models pins that this set matches the
# fields effective_format_version actually branches on, so a future hash-changing
# field added without an offset is caught.
HASH_CHANGING_FIELDS: tuple[str, ...] = ("freq_quantization", "window_bank", "image_mode")


def effective_format_version(config: FingerprintConfig) -> int:
    """Return the hash-derivation format version a ``config`` records.

    A default :class:`~fingerprint_engine.core.models.FingerprintConfig` -- and
    any config whose hash-changing fields are all at their defaults -- reports
    :data:`FINGERPRINT_FORMAT_VERSION` unchanged, so the stamped version is
    byte-identical to today for every existing index. Enabling an opt-in
    HASH-CHANGING flag (``freq_quantization`` > 1, a ``window_bank``, or
    ``image_mode == "phash"``) adds that flag's distinct offset, so a config that
    derives different hash codes reports a different version and an index built
    with it is *detectably* incompatible with a default index (see
    :meth:`HashIndex.search`). This is the mechanism that makes a future
    default-flip a deliberate version bump rather than a silent corpus
    corruption; it changes no hash code and no ranking itself.
    """

    version = FINGERPRINT_FORMAT_VERSION
    if config.freq_quantization > 1:
        version += _FORMAT_BUMP_FREQ_QUANTIZATION
    if config.window_bank:
        version += _FORMAT_BUMP_WINDOW_BANK
    if config.image_mode == "phash":
        version += _FORMAT_BUMP_IMAGE_PHASH
    return version


def coerce_format_version(raw: object, default: int = FINGERPRINT_FORMAT_VERSION) -> int:
    """Tolerantly coerce a stamped/persisted format version to an ``int``.

    The single rule behind the three readers of a recorded version
    (``Fingerprint.format_version`` over the config stamp, the snapshot reader,
    and the durable-backend reader): an ``int`` or an int-parseable ``str``
    yields that int; a ``bool``, a non-int/str, or an unparseable value falls
    back to ``default``. "Absent ⇒ default" keeps legacy data (written before the
    stamp existed) loadable and compatible rather than treated as a mismatch.
    """

    if isinstance(raw, bool) or not isinstance(raw, (int, str)):
        return default
    try:
        return int(raw)
    except ValueError:
        return default
