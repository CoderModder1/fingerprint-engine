"""Handler for archive/container files (zip/tar/tar.gz), stdlib-only.

Produces a *structural* fingerprint of an archive: a deterministic signal
derived from the sorted ``(member_name, member_size, member_digest)`` tuples of
its entries, so two archives with the same contents fingerprint similarly and an
archive with a single changed member is a near-duplicate (only that member's
contribution to the signal moves) while an archive with entirely different
contents is not.

Dependency-free: only the standard library ``zipfile``/``tarfile``/``hashlib``
is used, and member metadata (and, for small members, a content digest) is read
in-memory -- nothing is ever extracted to disk. Work on untrusted/hostile input
is bounded by four caps, on top of the ``max_file_size_bytes`` guard the
:class:`Fingerprinter` already enforces on the COMPRESSED size before the file is
ever opened: a file-member cap (``max_members``), a total-entry cap
(``max_entries``, so a tar flooded with non-file entries cannot iterate
unbounded), a per-member content-read cap (``max_member_content_bytes``), and an
AGGREGATE decompressed-byte budget (``max_total_content_bytes``) that bounds the
SUM of content reads -- the per-member cap alone still allowed ~``max_members`` x
``max_member_content_bytes`` (~4 GiB) of decompression, which the aggregate budget
closes. Exceeding the aggregate or traversal budget never raises: remaining
members fall back to their CRC/size identity token and the payload is flagged
``truncated``. Normal small archives cross none of these, so their fingerprint is
unchanged.
"""

from __future__ import annotations

import hashlib
import logging
import tarfile
import zipfile
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path

import numpy as np

from fingerprint_engine.core.models import FingerprintConfig

from .base import FileHandler

logger = logging.getLogger(__name__)

# Default cap on how many members are read from an archive. A deeply-nested or
# zip-bomb-style archive can declare millions of entries; reading their metadata
# is cheap individually but unbounded in aggregate, so the structural signal is
# built from at most this many (sorted) members. 0 = unlimited.
DEFAULT_MAX_ARCHIVE_MEMBERS = 4096

# Per-member byte cap for the optional content digest. Members at or below this
# size get a sha1 of their *content* mixed into the signal so two same-named,
# same-sized members with different bytes still differ; larger members fall back
# to their stored CRC/size only (reading them in full would reintroduce the OOM
# vector the size guard exists to prevent). 0 = never read content.
DEFAULT_MAX_MEMBER_CONTENT_BYTES = 1 << 20  # 1 MiB

# Default cap on TOTAL archive entries scanned (files + directories + symlinks +
# any other member type), distinct from ``max_members`` which caps only the FILE
# members whose identity enters the signal. It bounds header-driven iteration for
# a tar with a flood of non-file entries -- the file cap never increments for
# those, so without this an attacker-crafted tar could iterate unbounded. (A zip
# is already bounded by ``max_members`` over its central-directory index, so this
# is the tar-side guard.) 0 = unlimited.
DEFAULT_MAX_ARCHIVE_ENTRIES = 65536

# Default AGGREGATE budget for decompressed bytes read across ALL members for the
# content digest. The per-member cap above bounds ONE member's read; this bounds
# the SUM, so a high-ratio archive of many small members cannot force gigabytes of
# decompression (~max_members x 1 MiB ~= 4 GiB without it). Once the budget is
# exhausted, the remaining members fall back to their CRC/size identity token
# (never raising). For tar it ALSO caps how far the non-seekable gzip stream is
# traversed (advancing past a member decompresses its data), via the cumulative
# DECLARED member size -- see :meth:`ArchiveFileHandler._read_tar`. 0 = unlimited.
DEFAULT_MAX_TOTAL_CONTENT_BYTES = 256 << 20  # 256 MiB

# Each member contributes a fixed-length block of deterministic samples to the
# signal, so a one-member change perturbs only that block (a near-dup) rather
# than shifting every following sample (which a variable-length encoding would
# do). 256 samples/member keeps a handful of members comfortably above the fixed
# 512-sample window's frame floor while giving each block enough length to carry
# its own resolvable spectral peaks.
_SAMPLES_PER_MEMBER = 256

# Number of deterministic sinusoidal tones summed per member block. A small sum
# of member-derived tones produces a few clean, stable spectral peaks per member
# (rather than the flat spectrum of white noise), so an unchanged member's block
# yields the same constellation peaks at the same time positions in two archives
# -- which is what makes the shared members align and the changed one stand out.
_TONES_PER_MEMBER = 4


@dataclass(frozen=True)
class ArchiveMember:
    """One archive entry's stable identity (no extracted file bytes held)."""

    name: str
    size: int
    digest: str


@dataclass(frozen=True)
class ArchivePayload:
    archive_type: str
    members: list[ArchiveMember] = field(default_factory=list)
    truncated: bool = False


class ArchiveFileHandler(FileHandler):
    name = "archive"
    # Below image (60)/pdf (65)/audio (70) but above text (40): an archive is a
    # distinct container type and its extension/magic-byte signal drives routing,
    # so this only breaks ties -- it must out-rank the text/binary fallbacks for
    # a real archive while never shadowing the richer content handlers.
    priority = 55
    # Sequence-like structural signal: a small fixed window keeps archives of
    # different member counts comparable on a shared grid (same rationale as the
    # text/binary handlers), so a near-identical archive still aligns.
    default_signal_window = 512
    default_signal_hop = 128
    supported_mime_types = {
        "application/zip",
        "application/x-tar",
        "application/gzip",
        "application/x-gzip",
    }
    supported_extensions = {".zip", ".tar"}
    # Compound suffixes ``Path.suffix`` cannot see (it returns only ``.gz``).
    _compound_extensions = (".tar.gz", ".tgz")
    # Zip-based formats that are OWNED by a more specific handler and must not be
    # claimed as generic archives. ``.npz`` is a numpy vector container (a zip of
    # ``.npy`` members) routed to ``EmbeddingFileHandler``: it carries the zip
    # magic, so the magic sniff below would otherwise score it 0.90 and out-rank
    # the embedding handler's 0.80, producing a structural archive fingerprint
    # instead of the advertised vector-sequence one. Declining these by extension
    # keeps such files on their proper handler.
    _foreign_zip_extensions = (".npz",)

    def __init__(
        self,
        max_members: int | None = None,
        max_member_content_bytes: int | None = None,
        max_entries: int | None = None,
        max_total_content_bytes: int | None = None,
    ) -> None:
        if max_members is None:
            max_members = DEFAULT_MAX_ARCHIVE_MEMBERS
        if max_members < 0:
            raise ValueError("max_members must be non-negative (0 = unlimited)")
        if max_member_content_bytes is None:
            max_member_content_bytes = DEFAULT_MAX_MEMBER_CONTENT_BYTES
        if max_member_content_bytes < 0:
            raise ValueError("max_member_content_bytes must be non-negative (0 = never read content)")
        if max_entries is None:
            max_entries = DEFAULT_MAX_ARCHIVE_ENTRIES
        if max_entries < 0:
            raise ValueError("max_entries must be non-negative (0 = unlimited)")
        if max_total_content_bytes is None:
            max_total_content_bytes = DEFAULT_MAX_TOTAL_CONTENT_BYTES
        if max_total_content_bytes < 0:
            raise ValueError("max_total_content_bytes must be non-negative (0 = unlimited)")
        self.max_members = max_members
        self.max_member_content_bytes = max_member_content_bytes
        self.max_entries = max_entries
        self.max_total_content_bytes = max_total_content_bytes

    def _content_budget_remaining(self, used: int) -> int | None:
        """Remaining aggregate decompressed-read budget, or ``None`` if unlimited.

        ``None`` (``max_total_content_bytes == 0``) means no aggregate cap, so the
        digest helpers always read -- byte-identical to before the budget existed.
        """

        if not self.max_total_content_bytes:
            return None
        return max(0, self.max_total_content_bytes - used)

    @classmethod
    def can_handle(
        cls,
        path: str | Path,
        mime_type: str | None = None,
        sample: bytes | None = None,
    ) -> float:
        name = Path(path).name.lower()
        if any(name.endswith(ext) for ext in cls._foreign_zip_extensions):
            # A zip-based format owned by a more specific handler (e.g. ``.npz`` ->
            # embedding): decline so the magic sniff never out-ranks that handler.
            return 0.0
        if any(name.endswith(ext) for ext in cls._compound_extensions):
            # ``.tar.gz``/``.tgz`` are hidden from the base extension check
            # (suffix is only ``.gz``); credit them the same extension score.
            return max(super().can_handle(path, mime_type, sample), 0.75)
        base_score = super().can_handle(path, mime_type, sample)
        if base_score:
            return base_score
        if sample and cls._sniff_magic(sample) is not None:
            return 0.90
        return 0.0

    @staticmethod
    def _sniff_magic(sample: bytes) -> str | None:
        """Return an archive type from magic bytes, or ``None`` if not an archive.

        Gzip is reported as ``tar.gz`` on the assumption that a gzip stream is a
        compressed tar (the common case for these extensions); :meth:`load`
        verifies the real container and degrades gracefully if it is not.
        """

        if sample.startswith((b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")):
            return "zip"
        if sample.startswith(b"\x1f\x8b"):
            return "tar.gz"
        # POSIX tar: the "ustar" magic lives at offset 257 within the 512-byte
        # header, so it is only detectable when the sniff sample reaches that far.
        if len(sample) >= 262 and sample[257:262] == b"ustar":
            return "tar"
        return None

    def configure(self, config: FingerprintConfig) -> None:
        # The archive member cap is an additive, archive-only safety bound and is
        # not part of FingerprintConfig, so nothing to pull here. Defined so the
        # Fingerprinter's configure() call is an explicit, documented no-op for
        # this handler rather than relying on the base no-op.
        return

    def load(self, path: str | Path, *, content: bytes | None = None) -> ArchivePayload:
        source = Path(path)
        data = self.read_content(source, content)
        archive_type = self._classify(source, data)
        if archive_type == "zip":
            members, truncated = self._read_zip(data)
            return ArchivePayload("zip", members, truncated)
        # tar / tar.gz / unknown-gzip: tarfile sniffs the compression itself.
        members, truncated, resolved_type = self._read_tar(data, archive_type)
        return ArchivePayload(resolved_type, members, truncated)

    def _classify(self, source: Path, data: bytes) -> str:
        """Resolve the archive type from magic bytes, then extension as a hint."""

        sniffed = self._sniff_magic(data[:512])
        if sniffed is not None:
            return sniffed
        name = source.name.lower()
        if name.endswith(".zip"):
            return "zip"
        if any(name.endswith(ext) for ext in self._compound_extensions):
            return "tar.gz"
        if name.endswith(".tar"):
            return "tar"
        # No magic, no decisive extension: let tarfile try (it raises on a
        # non-archive, which load()'s caller turns into a handler fallback).
        return "tar"

    def _read_zip(self, data: bytes) -> tuple[list[ArchiveMember], bool]:
        members: list[ArchiveMember] = []
        truncated = False
        content_used = 0
        with zipfile.ZipFile(BytesIO(data)) as archive:
            infos = archive.infolist()
            # ``index`` counts every central-directory entry (dirs included), so
            # ``index >= max_members`` already bounds total zip iteration -- the
            # separate max_entries cap is the tar-side guard. Unchanged from before.
            for index, info in enumerate(infos):
                if self.max_members and index >= self.max_members:
                    truncated = True
                    break
                if info.is_dir():
                    continue
                remaining = self._content_budget_remaining(content_used)
                digest, read, budget_skipped = self._zip_member_digest(archive, info, remaining)
                content_used += read
                if budget_skipped:
                    # A member we'd normally content-digest fell back to its CRC
                    # token because the aggregate budget is exhausted -> coarser
                    # fingerprint, so flag truncation.
                    truncated = True
                members.append(ArchiveMember(info.filename, int(info.file_size), digest))
        return members, truncated

    def _zip_member_digest(
        self, archive: zipfile.ZipFile, info: zipfile.ZipInfo, budget_remaining: int | None
    ) -> tuple[str, int, bool]:
        """Content sha1 for small members (subject to the aggregate budget), else a CRC/size token.

        Returns ``(digest, bytes_read, budget_skipped)``. ``bytes_read`` is the
        decompressed bytes actually read, accumulated by the caller against the
        aggregate :attr:`max_total_content_bytes` budget; ``budget_skipped`` is
        ``True`` when a member that WOULD be content-digested instead got the CRC
        fallback because the budget is exhausted. For a normal in-budget archive
        every small member is read in full, so the returned digest -- and thus the
        fingerprint -- is byte-identical to before the budget existed. The CRC-32
        is already in the central directory, so the large-member/over-budget
        fallback costs no extra read while still distinguishing same-named entries.
        """

        if self.max_member_content_bytes and 0 < info.file_size <= self.max_member_content_bytes:
            to_read = min(int(info.file_size), self.max_member_content_bytes)
            if budget_remaining is not None and to_read > budget_remaining:
                return f"crc:{info.CRC:08x}:{info.file_size}", 0, True
            try:
                with archive.open(info) as handle:
                    # Read exactly to_read (== member size for an eligible member),
                    # not the raw cap: byte-identical output, but it never decodes
                    # past the member (matters for a non-seekable gzip-tar stream).
                    data = handle.read(to_read)
                    return hashlib.sha1(data).hexdigest(), len(data), False
            except Exception as exc:  # noqa: BLE001 - degrade to CRC identity, never abort
                logger.debug("zip member %s content read failed: %s", info.filename, exc)
        return f"crc:{info.CRC:08x}:{info.file_size}", 0, False

    def _read_tar(
        self, data: bytes, archive_type: str
    ) -> tuple[list[ArchiveMember], bool, str]:
        members: list[ArchiveMember] = []
        truncated = False
        # 'r:*' lets tarfile transparently sniff gzip/bzip2/xz/uncompressed, so a
        # mislabeled ``.tar.gz`` that is really a plain tar (or vice versa) still
        # reads; the resolved type is reported back for metadata.
        content_used = 0
        declared_total = 0
        with tarfile.open(fileobj=BytesIO(data), mode="r:*") as archive:
            total_entries = 0
            file_count = 0
            for member in archive:
                # Count EVERY entry (dirs/symlinks/devices/...) toward the entry
                # cap, so a tar flooded with non-file entries cannot iterate
                # unbounded -- the prior code only counted regular files, so the
                # file cap never bounded that iteration.
                total_entries += 1
                if self.max_entries and total_entries > self.max_entries:
                    truncated = True
                    break
                if not member.isfile():
                    continue
                if self.max_members and file_count >= self.max_members:
                    truncated = True
                    break
                remaining = self._content_budget_remaining(content_used)
                digest, read, budget_skipped = self._tar_member_digest(archive, member, remaining)
                content_used += read
                members.append(ArchiveMember(member.name, int(member.size), digest))
                file_count += 1
                if budget_skipped:
                    truncated = True
                # Bound how far we traverse the (non-seekable, gzip) stream:
                # advancing PAST a member decompresses its data, so a few huge
                # members could otherwise force gigabytes of skip-decompression.
                # Stop once the cumulative DECLARED member size exceeds the budget;
                # breaking here means tarfile never advances past (decompresses)
                # the member that tripped it. Normal small tars never reach this.
                declared_total += max(0, int(member.size))
                if self.max_total_content_bytes and declared_total >= self.max_total_content_bytes:
                    truncated = True
                    break
        return members, truncated, archive_type

    def _tar_member_digest(
        self, archive: tarfile.TarFile, member: tarfile.TarInfo, budget_remaining: int | None
    ) -> tuple[str, int, bool]:
        """Content sha1 for small members (subject to the aggregate budget), else a size token.

        Returns ``(digest, bytes_read, budget_skipped)`` like the zip variant.
        Unlike zip, tar headers carry no CRC, so a large/over-budget member falls
        back to its size only -- still stable, just coarser. For a normal in-budget
        archive every small member is read in full, so the digest is byte-identical
        to before the budget existed.
        """

        if self.max_member_content_bytes and 0 < member.size <= self.max_member_content_bytes:
            to_read = min(int(member.size), self.max_member_content_bytes)
            if budget_remaining is not None and to_read > budget_remaining:
                return f"sz:{member.size}", 0, True
            try:
                handle = archive.extractfile(member)
                if handle is not None:
                    with handle:
                        # Read exactly to_read (== member size), not the raw cap, so
                        # a tiny member never forces the non-seekable gzip stream to
                        # decode ahead into the next member. Byte-identical output.
                        data = handle.read(to_read)
                        return hashlib.sha1(data).hexdigest(), len(data), False
            except Exception as exc:  # noqa: BLE001 - degrade to size identity, never abort
                logger.debug("tar member %s content read failed: %s", member.name, exc)
        return f"sz:{member.size}", 0, False

    def to_signal(self, payload: ArchivePayload) -> np.ndarray:
        if not payload.members:
            return np.zeros(1, dtype=np.float32)

        # Sort by the full stable identity so member *order* in the archive does
        # not affect the fingerprint -- only the set of (name, size, digest)
        # tuples does. Each member then deterministically expands into a fixed
        # block of samples, so changing one member perturbs only its block.
        ordered = sorted(
            payload.members,
            key=lambda m: (m.name, m.size, m.digest),
        )
        blocks = [self._member_block(member) for member in ordered]
        return np.concatenate(blocks).astype(np.float32, copy=False)

    @staticmethod
    def _member_block(member: ArchiveMember) -> np.ndarray:
        """Deterministic fixed-length tonal block for one member.

        The block is a sum of ``_TONES_PER_MEMBER`` sinusoids whose frequencies,
        amplitudes, and phases are derived from a sha256 of the member's stable
        identity, so the block is a stable function of ``(name, size, digest)``
        and independent of every other member. A tonal block (rather than white
        noise) produces a few clean, repeatable spectral peaks, so an unchanged
        member yields the *same* constellation peaks in two archives -- the
        property that makes shared members align and a single changed member
        stand out as a near-dup.
        """

        identity = f"{member.name}\x00{member.size}\x00{member.digest}".encode()
        seed = hashlib.sha256(identity).digest()
        positions = np.arange(_SAMPLES_PER_MEMBER, dtype=np.float64)
        block = np.zeros(_SAMPLES_PER_MEMBER, dtype=np.float64)
        for tone in range(_TONES_PER_MEMBER):
            base = tone * 6
            # Three independent bytes per tone -> frequency, amplitude, phase.
            freq = 1.0 + (seed[base] / 255.0) * (_SAMPLES_PER_MEMBER / 4.0)
            amplitude = 0.25 + (seed[base + 1] / 255.0) * 0.75
            phase = (seed[base + 2] / 255.0) * 2.0 * np.pi
            block += amplitude * np.sin(2.0 * np.pi * freq * positions / _SAMPLES_PER_MEMBER + phase)
        # Center in [-1, 1] to sit in the same numeric range as the other
        # handlers' signals before the pipeline's own normalization.
        peak = float(np.max(np.abs(block)))
        if peak > 0.0:
            block = block / peak
        return block

    def metadata(self, payload: ArchivePayload) -> dict[str, object]:
        return {
            "archive_type": payload.archive_type,
            "member_count": len(payload.members),
            "members_truncated": payload.truncated,
            "signal_strategy": "sorted_member_identity_blocks",
        }
