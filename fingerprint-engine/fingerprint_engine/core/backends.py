"""Index-backend construction shared by the CLI and the HTTP service.

Both entry points select one of the four backends from user configuration -- the
CLI from argparse flags, the service from ``FINGERPRINT_*`` environment variables
-- and otherwise build the same objects. :func:`open_backend` is that single
construction point, and the ``DEFAULT_*`` constants are the one source of truth
for the connection defaults (the CLI flag defaults and the service env fallbacks
both reference them, so they cannot drift apart).
"""

from __future__ import annotations

from pathlib import Path

from .index import (
    HashIndex,
    InMemoryHashIndex,
    PostgresHashIndex,
    RedisHashIndex,
    SQLiteHashIndex,
)

DEFAULT_SQLITE_PATH = ".fingerprint_index.sqlite3"
DEFAULT_REDIS_URL = "redis://localhost:6379/0"
DEFAULT_REDIS_PREFIX = "fpidx"
DEFAULT_POSTGRES_DSN = "postgresql://localhost/fingerprint"

# The backends a caller may select, for validation/messaging by the entry points.
BACKEND_CHOICES = ("memory", "sqlite", "redis", "postgres")


def open_backend(
    backend: str,
    *,
    sqlite_path: str = DEFAULT_SQLITE_PATH,
    redis_url: str = DEFAULT_REDIS_URL,
    redis_prefix: str = DEFAULT_REDIS_PREFIX,
    postgres_dsn: str = DEFAULT_POSTGRES_DSN,
    memory_path: str | Path | None = None,
) -> HashIndex:
    """Construct the selected index backend.

    The one asymmetry between the CLI and the service is the in-memory backend:
    the CLI persists to a JSON file and so LOADS it (``memory_path`` given), while
    the service starts fresh (``memory_path is None``). Redis/SQLite/Postgres are
    constructed identically for both. An unrecognized ``backend`` raises
    :class:`ValueError`; callers that want a configuration-specific message
    (e.g. naming the env var) should validate against :data:`BACKEND_CHOICES`
    first.
    """

    backend = (backend or "memory").lower()
    if backend == "redis":
        return RedisHashIndex(url=redis_url, key_prefix=redis_prefix)
    if backend == "sqlite":
        return SQLiteHashIndex(database=sqlite_path)
    if backend == "postgres":
        return PostgresHashIndex(dsn=postgres_dsn)
    if backend == "memory":
        return InMemoryHashIndex.load(Path(memory_path)) if memory_path is not None else InMemoryHashIndex()
    raise ValueError(f"unknown backend {backend!r}; expected one of {', '.join(BACKEND_CHOICES)}")
