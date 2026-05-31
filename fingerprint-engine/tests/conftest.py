"""Shared pytest fixtures.

The auto-injected fixtures live here (pytest discovers conftest automatically);
the importable helper FUNCTIONS and the ``requires_pg`` marker live in
``_fixtures.py`` (imported by name). ``pythonpath = ["."]`` in pyproject puts the
package on sys.path, so test modules no longer need per-file sys.path bootstraps.
"""

from __future__ import annotations

import pytest
from _fixtures import PG_DSN

from fingerprint_engine.core.index import PostgresHashIndex


@pytest.fixture
def pg_index():  # noqa: ANN201 - test fixture
    """A clean Postgres-backed index against the live server in FINGERPRINT_TEST_PG_DSN.

    Truncates on entry and drops its tables on exit, so @requires_pg tests start
    from a clean slate and leave nothing behind. Skipped when the DSN is unset.
    """

    index = PostgresHashIndex(dsn=PG_DSN, table_prefix="fp_pytest")
    with index._conn.cursor() as cur:  # start from a clean slate
        cur.execute(f"TRUNCATE {index._files_table}, {index._postings_table}")
    index._conn.commit()
    try:
        yield index
    finally:
        with index._conn.cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS {index._postings_table}, {index._files_table}")
        index._conn.commit()
        index.close()
