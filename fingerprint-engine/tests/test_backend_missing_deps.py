"""The SQL/Redis backends must raise ``MissingDependencyError`` (not a bare
``RuntimeError``) when their optional driver is absent -- the documented
lazy-extras contract, so a caller catching ``MissingDependencyError`` handles
every extra uniformly and the CLI maps it to exit code 3 (missing dependency).

These exercise the ``except ImportError`` branch in each backend constructor,
which is otherwise ``# pragma: no cover`` because the driver is installed in dev.
"""

from __future__ import annotations

import builtins
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fingerprint_engine.core.exceptions import MissingDependencyError
from fingerprint_engine.core.index import PostgresHashIndex, RedisHashIndex


def _block_top_level_import(monkeypatch: pytest.MonkeyPatch, blocked: str) -> None:
    """Make ``import <blocked>`` raise ImportError even if it is installed."""

    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == blocked or name.startswith(f"{blocked}."):
            raise ImportError(f"No module named {blocked!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)


def test_redis_backend_missing_driver_raises_missing_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _block_top_level_import(monkeypatch, "redis")
    with pytest.raises(MissingDependencyError) as excinfo:
        RedisHashIndex(url="redis://localhost:6379/0")
    assert excinfo.value.package == "redis"
    assert excinfo.value.extra == "redis"


def test_postgres_backend_missing_driver_raises_missing_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _block_top_level_import(monkeypatch, "psycopg")
    with pytest.raises(MissingDependencyError) as excinfo:
        PostgresHashIndex(dsn="postgresql://localhost/does_not_matter")
    assert excinfo.value.package == "psycopg"
    assert excinfo.value.extra == "postgres"
