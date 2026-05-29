"""Exception hierarchy for fingerprinting failures."""

from __future__ import annotations


class FingerprintError(Exception):
    """Base class for fingerprinting failures."""


class NoHandlerError(FingerprintError):
    """No handler was able to fingerprint the file."""


class InvalidSnapshotError(FingerprintError, ValueError):
    """A persisted index snapshot is structurally invalid or unsupported.

    Multiply inherits from :class:`ValueError` so callers that already guard
    snapshot loads with ``except ValueError`` keep catching it, while it also
    joins the :class:`FingerprintError` family for unified error handling.
    Raised for malformed structure (e.g. ``files`` not a mapping), a corrupt
    primary/backup snapshot, or an unsupported ``schema_version``.
    """


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
