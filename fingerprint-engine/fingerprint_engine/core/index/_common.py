"""Index-internal helpers: snapshot/format-version utilities, the per-index
lock decorator, and the per-file metadata builder shared by the backends."""

from __future__ import annotations

import functools
import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar, cast

from ..exceptions import (
    InvalidSnapshotError,
)
from ..models import (
    FINGERPRINT_FORMAT_VERSION,
    Fingerprint,
)

logger = logging.getLogger(__name__)

# Schema version stamped into every snapshot written by :meth:`HashIndex.save`.
# A snapshot whose top-level ``schema_version`` is present but not in
# ``_SUPPORTED_SCHEMA_VERSIONS`` is rejected on load; an ABSENT version is
# treated as version 1 for backward compatibility with already-written files.
SNAPSHOT_SCHEMA_VERSION = 1
_SUPPORTED_SCHEMA_VERSIONS = frozenset({1})

# Top-level snapshot field recording the HASH-DERIVATION format version of the
# indexed postings (see ``FINGERPRINT_FORMAT_VERSION`` in ``core/models.py``).
# DISTINCT from ``schema_version``: schema versions the JSON container, this
# versions the meaning of the ``hash_code`` integers inside it. An ABSENT field
# is treated as the default ``FINGERPRINT_FORMAT_VERSION`` (legacy snapshots
# written before the field existed stay loadable and compatible).
SNAPSHOT_FORMAT_VERSION_KEY = "fingerprint_format_version"

# Hash codes are unsigned 64-bit; reject snapshot postings outside this range
# before they reach a SQL backend's signed-offset encode (which would raise
# OverflowError deep in the load and abort the whole cross-backend import).
_HASH_CODE_MIN = 0
_HASH_CODE_MAX = (1 << 64) - 1


def _validate_schema_version(data: dict[str, object]) -> None:
    """Raise :class:`InvalidSnapshotError` for an unsupported snapshot version.

    An absent ``schema_version`` is treated as version 1 (legacy snapshots
    written before versioning was introduced remain loadable).
    """

    if "schema_version" not in data:
        return
    version = data["schema_version"]
    if version not in _SUPPORTED_SCHEMA_VERSIONS:
        raise InvalidSnapshotError(
            f"unsupported snapshot schema_version {version!r}; "
            f"this build supports {sorted(_SUPPORTED_SCHEMA_VERSIONS)}"
        )


def _snapshot_format_version(data: dict[str, object]) -> int:
    """Read a snapshot's hash-derivation format version.

    An ABSENT :data:`SNAPSHOT_FORMAT_VERSION_KEY` is treated as the default
    :data:`FINGERPRINT_FORMAT_VERSION` -- legacy snapshots written before the
    field existed describe the default derivation, so they stay loadable and
    compatible. A malformed (non-int) value also falls back to the default
    rather than aborting the load; the version is advisory metadata for the
    search-time compatibility check, never a gate on loadability.
    """

    raw = data.get(SNAPSHOT_FORMAT_VERSION_KEY, FINGERPRINT_FORMAT_VERSION)
    if isinstance(raw, bool) or not isinstance(raw, (int, str)):
        return FINGERPRINT_FORMAT_VERSION
    try:
        return int(raw)
    except ValueError:
        return FINGERPRINT_FORMAT_VERSION


def _in_hash_range(hash_code: int) -> bool:
    """Whether ``hash_code`` fits the unsigned 64-bit posting range."""

    return _HASH_CODE_MIN <= hash_code <= _HASH_CODE_MAX


def _fsync_path(path: Path) -> None:
    """Best-effort fsync of a file's data blocks (never raises on an unsupported FS).

    Used to make a freshly written backup durable before the primary is
    replaced. Like the parent-directory fsync, this is best-effort: not every
    platform/filesystem supports it, so a failure is swallowed rather than
    allowed to break the save.
    """

    try:
        fd = os.open(str(path), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


def _index_metadata(fingerprint: Fingerprint) -> dict[str, Any]:
    """The per-file metadata dict every backend stores for a fingerprint.

    Single source of truth for the index-metadata shape, so all four backends
    store an identical record -- which is what makes list_files/iter_metadata and
    snapshot interop parity hold. ``**fingerprint.metadata`` is merged LAST so a
    handler's extra keys never shadow the canonical fields. The SQL/Redis backends
    json.dumps(..., sort_keys=True) it (so insertion order is normalized there);
    InMemory stores it directly and to_dict serializes it.
    """

    return {
        "file_id": fingerprint.file_id,
        "path": fingerprint.path,
        "handler": fingerprint.handler,
        "size_bytes": fingerprint.size_bytes,
        "content_sha256": fingerprint.content_sha256,
        "hash_count": fingerprint.hash_count,
        "landmark_count": fingerprint.landmark_count,
        **fingerprint.metadata,
    }


_BackendMethod = TypeVar("_BackendMethod", bound=Callable[..., Any])


def _synchronized(method: _BackendMethod) -> _BackendMethod:
    """Run a backend method while holding ``self._lock`` (the per-index RLock).

    Used by the SQL backends (F3): a single shared DB connection is one serial
    command stream, so concurrent FastAPI-threadpool requests would otherwise
    interleave the multi-statement critical sections -- the SQLite ``_query`` temp
    table, the write transactions -- and corrupt results or raise ``InterfaceError``.
    Serializing every connection-touching method closes that race. The lock is
    re-entrant (``RLock``), so a method that calls another synchronized method
    (``add`` -> ``remove``, ``to_dict`` -> ``_metadata_for``) never self-deadlocks,
    and it is uncontended on the single-threaded path so output is byte-identical.
    """

    @functools.wraps(method)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        with self._lock:
            return method(self, *args, **kwargs)

    return cast(_BackendMethod, wrapper)
