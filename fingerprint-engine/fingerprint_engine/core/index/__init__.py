"""Searchable fingerprint index backends.

Split from the former monolithic core/index.py into a package; this __init__
preserves ``from fingerprint_engine.core.index import ...`` for every prior
public name (the base + four backends and the two snapshot/format-version
constants tests import by name)."""

from ._common import SNAPSHOT_FORMAT_VERSION_KEY, SNAPSHOT_SCHEMA_VERSION
from .base import HashIndex
from .memory import InMemoryHashIndex
from .postgres import PostgresHashIndex
from .redis import RedisHashIndex
from .sqlite import SQLiteHashIndex

__all__ = [
    "HashIndex",
    "InMemoryHashIndex",
    "PostgresHashIndex",
    "RedisHashIndex",
    "SQLiteHashIndex",
    "SNAPSHOT_FORMAT_VERSION_KEY",
    "SNAPSHOT_SCHEMA_VERSION",
]
