"""Exception hierarchy for fingerprinting failures."""

from __future__ import annotations


class FingerprintError(Exception):
    """Base class for fingerprinting failures."""


class NoHandlerError(FingerprintError):
    """No handler was able to fingerprint the file."""


class FileTooLargeError(FingerprintError):
    """An input file exceeds the configured ``max_file_size_bytes`` limit.

    Raised by :meth:`Fingerprinter.fingerprint_file` *before* the whole file is
    read into memory, so an oversized (potentially malicious) input never gets
    loaded. Carries the offending ``size`` and the configured ``limit`` (both in
    bytes) for clear diagnostics.
    """

    def __init__(self, message: str, *, size: int, limit: int) -> None:
        super().__init__(message)
        self.size = size
        self.limit = limit


class InvalidSnapshotError(FingerprintError, ValueError):
    """A persisted index snapshot is structurally invalid or unsupported.

    Multiply inherits from :class:`ValueError` so callers that already guard
    snapshot loads with ``except ValueError`` keep catching it, while it also
    joins the :class:`FingerprintError` family for unified error handling.
    Raised for malformed structure (e.g. ``files`` not a mapping), a corrupt
    primary/backup snapshot, or an unsupported ``schema_version``.
    """


class FormatVersionMismatchError(FingerprintError):
    """A query was searched against an index built with a different hash format.

    Hash codes carry no meaning across hash-derivation formats (see
    ``FINGERPRINT_FORMAT_VERSION`` in ``core/models.py``): a query fingerprinted
    under one format and an index built under another do not share a code space,
    so any "match" between them is a false result, not a weak one. The default
    :meth:`HashIndex.search` only *warns* (a :class:`RuntimeWarning`) on a
    mismatch so an existing pipeline is never broken by the new check; callers
    that want a hard failure pass ``strict_format=True`` to raise this instead.
    Carries the offending ``query_version`` and ``index_version``.
    """

    def __init__(self, message: str, *, query_version: int, index_version: int) -> None:
        super().__init__(message)
        self.query_version = query_version
        self.index_version = index_version


class MissingDependencyError(FingerprintError):
    """The correct handler exists but its optional dependency is not installed.

    Raised instead of silently demoting to a lower-priority handler (e.g. the
    binary fallback), which would produce fingerprints incomparable to those
    made where the dependency is installed and silently corrupt the index.
    """

    def __init__(
        self,
        message: str,
        *,
        package: str | None = None,
        extra: str | None = None,
    ) -> None:
        super().__init__(message)
        self.package = package
        self.extra = extra
