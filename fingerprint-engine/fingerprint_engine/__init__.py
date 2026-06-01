"""Universal file fingerprinting engine.

Public API. Importing this package pulls in only the core orchestration,
FFT pipeline, and index contract (which depend on ``numpy`` alone). Optional
handlers and backends lazy-import their extras (Pillow, scipy, pydub, pypdf,
redis, psycopg) on first use, so this top-level import stays cheap and never
fails for a core-only install.
"""

from __future__ import annotations

import logging
from importlib.metadata import PackageNotFoundError, version

from .core.dedup import (
    DedupReport,
    ExactDuplicateCluster,
    NearDuplicateCluster,
    find_duplicates,
)
from .core.exceptions import (
    FingerprintError,
    FormatVersionMismatchError,
    InvalidSnapshotError,
    MissingDependencyError,
    NoHandlerError,
    SnapshotWriteRefused,
)
from .core.fingerprinter import Fingerprinter
from .core.index import (
    HashIndex,
    InMemoryHashIndex,
    PostgresHashIndex,
    RedisHashIndex,
    SQLiteHashIndex,
)
from .core.models import (
    FINGERPRINT_FORMAT_VERSION,
    Calibration,
    ConstellationHash,
    Fingerprint,
    FingerprintConfig,
    IndexPosting,
    LandmarkPoint,
    SearchResult,
    effective_format_version,
)

try:
    __version__ = version("fingerprint-engine")
except PackageNotFoundError:  # pragma: no cover - running from a source tree
    __version__ = "1.0.0"

# Library logging best practice: attach a NullHandler to the package's top-level
# logger so emitting records never triggers Python's "No handlers could be found"
# warning when the embedding application has not configured logging. We do NOT
# call logging.basicConfig or attach any real handler -- the application owns
# handler/level configuration; per-module loggers roll up under this parent.
logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "FINGERPRINT_FORMAT_VERSION",
    "Calibration",
    "ConstellationHash",
    "DedupReport",
    "ExactDuplicateCluster",
    "Fingerprint",
    "FingerprintConfig",
    "FingerprintError",
    "Fingerprinter",
    "FormatVersionMismatchError",
    "HashIndex",
    "InMemoryHashIndex",
    "IndexPosting",
    "InvalidSnapshotError",
    "LandmarkPoint",
    "MissingDependencyError",
    "NearDuplicateCluster",
    "NoHandlerError",
    "PostgresHashIndex",
    "RedisHashIndex",
    "SQLiteHashIndex",
    "SearchResult",
    "SnapshotWriteRefused",
    "__version__",
    "effective_format_version",
    "find_duplicates",
]
