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

from .core.exceptions import (
    FingerprintError,
    InvalidSnapshotError,
    MissingDependencyError,
    NoHandlerError,
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
    Calibration,
    ConstellationHash,
    Fingerprint,
    FingerprintConfig,
    LandmarkPoint,
    SearchResult,
)

try:
    __version__ = version("fingerprint-engine")
except PackageNotFoundError:  # pragma: no cover - running from a source tree
    __version__ = "0.1.0"

# Library logging best practice: attach a NullHandler to the package's top-level
# logger so emitting records never triggers Python's "No handlers could be found"
# warning when the embedding application has not configured logging. We do NOT
# call logging.basicConfig or attach any real handler -- the application owns
# handler/level configuration; per-module loggers roll up under this parent.
logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "Calibration",
    "ConstellationHash",
    "Fingerprint",
    "FingerprintConfig",
    "FingerprintError",
    "Fingerprinter",
    "HashIndex",
    "InMemoryHashIndex",
    "InvalidSnapshotError",
    "LandmarkPoint",
    "MissingDependencyError",
    "NoHandlerError",
    "PostgresHashIndex",
    "RedisHashIndex",
    "SQLiteHashIndex",
    "SearchResult",
    "__version__",
]
