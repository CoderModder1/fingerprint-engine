"""Tests for the package's public surface and import hygiene.

These guard two invariants that are easy to break silently when the package
layout or dependency boundaries change:

1. Importing the top-level package must not pull in any optional dependency
   (Pillow/scipy/pydub/pypdf/redis/psycopg). They are lazily imported only by
   the handler/backend that needs them, so a core-only install stays light.
2. Every name advertised in ``__all__`` must actually resolve on the package,
   and ``__version__`` must be exposed.
"""

from __future__ import annotations

import subprocess
import sys

OPTIONAL_DEPS = (
    "PIL",
    "scipy",
    "pydub",
    "pypdf",
    "av",
    "sentence_transformers",
    "redis",
    "psycopg",
)


def test_importing_package_pulls_no_optional_dependency() -> None:
    # Run in a FRESH interpreter via subprocess: that is the only correct way to
    # test "does importing X pull in Y" and it avoids mutating this process's
    # sys.modules (deleting fingerprint_engine.* here would corrupt the import-
    # time handler registry that other tests rely on, making the suite order-
    # dependent).
    code = (
        "import importlib, sys\n"
        "importlib.import_module('fingerprint_engine')\n"
        f"deps = {OPTIONAL_DEPS!r}\n"
        "leaked = [d for d in deps if d in sys.modules]\n"
        "assert not leaked, 'optional deps leaked at import: ' + repr(leaked)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, (
        f"importing fingerprint_engine pulled in optional deps.\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )


def test_all_exports_resolve_and_version_present() -> None:
    import fingerprint_engine as fe

    assert isinstance(fe.__version__, str) and fe.__version__
    assert fe.__all__, "__all__ should not be empty"
    missing = [name for name in fe.__all__ if not hasattr(fe, name)]
    assert missing == [], f"__all__ names without an attribute: {missing}"
