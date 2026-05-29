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

import importlib
import sys

OPTIONAL_DEPS = ("PIL", "scipy", "pydub", "pypdf", "redis", "psycopg")


def test_importing_package_pulls_no_optional_dependency() -> None:
    # Drop any optional deps a prior test may have imported, then re-import the
    # package fresh and confirm none of them got dragged in transitively.
    for name in (*OPTIONAL_DEPS, "fingerprint_engine"):
        for mod in [m for m in sys.modules if m == name or m.startswith(name + ".")]:
            del sys.modules[mod]

    importlib.import_module("fingerprint_engine")

    leaked = [name for name in OPTIONAL_DEPS if name in sys.modules]
    assert leaked == [], f"importing fingerprint_engine pulled in optional deps: {leaked}"


def test_all_exports_resolve_and_version_present() -> None:
    import fingerprint_engine as fe

    assert isinstance(fe.__version__, str) and fe.__version__
    assert fe.__all__, "__all__ should not be empty"
    missing = [name for name in fe.__all__ if not hasattr(fe, name)]
    assert missing == [], f"__all__ names without an attribute: {missing}"
