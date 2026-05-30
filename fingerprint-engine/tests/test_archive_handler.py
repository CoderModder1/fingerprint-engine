"""Tests for the stdlib-only archive/container handler.

Covers routing (``can_handle`` for archives vs. non-archives), structural
fingerprinting (non-zero hashes, identical-content match, one-member near-dup
vs. totally-different non-match), magic-byte recognition, the member cap, and
graceful handling of malformed/empty archives.
"""

from __future__ import annotations

import io
import tarfile
import zipfile
from pathlib import Path

import pytest

from fingerprint_engine.core.fingerprinter import Fingerprinter
from fingerprint_engine.core.index import InMemoryHashIndex
from fingerprint_engine.handlers.archive_handler import (
    DEFAULT_MAX_ARCHIVE_ENTRIES,
    DEFAULT_MAX_ARCHIVE_MEMBERS,
    DEFAULT_MAX_TOTAL_CONTENT_BYTES,
    ArchiveFileHandler,
)


def _make_zip(path: Path, members: list[tuple[str, bytes]]) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in members:
            archive.writestr(name, data)


def _make_targz(path: Path, members: list[tuple[str, bytes]]) -> None:
    with tarfile.open(path, "w:gz") as archive:
        for name, data in members:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))


def _base_members(count: int = 8) -> list[tuple[str, bytes]]:
    # Each member is large enough that its small-member content digest is read,
    # so a byte change in one member changes that member's identity.
    return [(f"file{i}.txt", (f"content number {i} " * 80).encode()) for i in range(count)]


def _hash_overlap(a: object, b: object) -> float:
    sa = {h.hash_code for h in a.hashes}  # type: ignore[attr-defined]
    sb = {h.hash_code for h in b.hashes}  # type: ignore[attr-defined]
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


# ---------------------------------------------------------------------------
# Routing: archives score, non-archives score 0 (default-preserving)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "mime"),
    [
        ("a.zip", "application/zip"),
        ("a.tar", "application/x-tar"),
        ("a.tar.gz", "application/x-tar"),
        ("a.tgz", "application/x-tar"),
    ],
)
def test_can_handle_recognizes_archive_extensions(name: str, mime: str) -> None:
    assert ArchiveFileHandler.can_handle(name, mime_type=mime, sample=None) > 0.0


@pytest.mark.parametrize(
    ("name", "mime", "sample"),
    [
        ("script.py", "text/x-python", b"import os\n"),
        ("image.png", "image/png", b"\x89PNG\r\n\x1a\n"),
        ("note.txt", "text/plain", b"hello world"),
        ("data.json", "application/json", b"{}"),
        ("clip.wav", "audio/wav", b"RIFF"),
    ],
)
def test_can_handle_returns_zero_for_non_archives(
    name: str, mime: str, sample: bytes
) -> None:
    # The new handler must not change routing for non-archive files: it scores
    # 0.0 so it never out-ranks (or even competes with) the existing handlers.
    assert ArchiveFileHandler.can_handle(name, mime_type=mime, sample=sample) == 0.0


def test_can_handle_recognizes_zip_by_magic_without_extension(tmp_path: Path) -> None:
    path = tmp_path / "mystery"  # no archive extension at all
    _make_zip(path, _base_members())
    sample = path.read_bytes()[:512]
    assert ArchiveFileHandler.can_handle(path, mime_type=None, sample=sample) >= 0.90


def test_can_handle_recognizes_gzip_by_magic(tmp_path: Path) -> None:
    path = tmp_path / "mystery"
    _make_targz(path, _base_members())
    sample = path.read_bytes()[:512]
    assert ArchiveFileHandler.can_handle(path, mime_type=None, sample=sample) >= 0.90


# ---------------------------------------------------------------------------
# Structural fingerprinting via the real Fingerprinter
# ---------------------------------------------------------------------------


def test_zip_fingerprints_with_nonzero_hashes(tmp_path: Path) -> None:
    path = tmp_path / "a.zip"
    _make_zip(path, _base_members())

    fingerprint = Fingerprinter().fingerprint_file(path)

    assert fingerprint.handler == "archive"
    assert fingerprint.hash_count > 0
    assert fingerprint.metadata["archive_type"] == "zip"
    assert fingerprint.metadata["member_count"] == 8


def test_identical_content_archives_fingerprint_identically(tmp_path: Path) -> None:
    members = _base_members()
    p1 = tmp_path / "one.zip"
    p2 = tmp_path / "two.zip"
    _make_zip(p1, members)
    _make_zip(p2, members)

    fp = Fingerprinter()
    f1 = fp.fingerprint_file(p1)
    f2 = fp.fingerprint_file(p2)

    # Same member set -> byte-identical structural hash set.
    assert set(f1.hash_tuples()) == set(f2.hash_tuples())


def test_member_order_does_not_change_fingerprint(tmp_path: Path) -> None:
    members = _base_members(6)
    p1 = tmp_path / "order_a.zip"
    p2 = tmp_path / "order_b.zip"
    _make_zip(p1, members)
    _make_zip(p2, list(reversed(members)))

    fp = Fingerprinter()
    f1 = fp.fingerprint_file(p1)
    f2 = fp.fingerprint_file(p2)

    # The signal sorts members by identity, so archive order is irrelevant.
    assert set(f1.hash_tuples()) == set(f2.hash_tuples())


def test_one_changed_member_is_near_dup_but_different_archive_is_not(
    tmp_path: Path,
) -> None:
    members = _base_members()
    original = tmp_path / "original.zip"
    _make_zip(original, members)

    near = list(members)
    near[3] = ("file3.txt", b"COMPLETELY DIFFERENT bytes here " * 80)
    near_path = tmp_path / "near.zip"
    _make_zip(near_path, near)

    different = [(f"other{i}.dat", (f"unrelated {i} " * 80).encode()) for i in range(8)]
    different_path = tmp_path / "different.zip"
    _make_zip(different_path, different)

    fp = Fingerprinter()
    f_orig = fp.fingerprint_file(original)
    f_near = fp.fingerprint_file(near_path)
    f_diff = fp.fingerprint_file(different_path)

    near_overlap = _hash_overlap(f_orig, f_near)
    diff_overlap = _hash_overlap(f_orig, f_diff)

    # A single changed member out of eight keeps most blocks intact -> high
    # overlap; a wholly different member set shares no structural hashes.
    assert near_overlap > 0.2
    assert diff_overlap == 0.0
    assert near_overlap > diff_overlap

    # And the same separation shows up in an actual index search.
    index = InMemoryHashIndex()
    index.add(f_orig)
    near_results = index.search(f_near, top_k=5)
    diff_results = index.search(f_diff, top_k=5)
    assert near_results, "one-member-changed archive should match in the index"
    assert near_results[0].file_id == f_orig.file_id
    assert not diff_results, "a totally different archive should not match"


def test_targz_fingerprints_like_zip(tmp_path: Path) -> None:
    path = tmp_path / "arch.tar.gz"
    _make_targz(path, _base_members())

    fingerprint = Fingerprinter().fingerprint_file(path)

    assert fingerprint.handler == "archive"
    assert fingerprint.hash_count > 0
    assert fingerprint.metadata["archive_type"] == "tar.gz"
    assert fingerprint.metadata["member_count"] == 8


# ---------------------------------------------------------------------------
# Guards: member cap + malformed/empty handling
# ---------------------------------------------------------------------------


def test_member_cap_bounds_members_read(tmp_path: Path) -> None:
    path = tmp_path / "many.zip"
    _make_zip(path, _base_members(10))

    payload = ArchiveFileHandler(max_members=3).load(path)
    assert len(payload.members) == 3
    assert payload.truncated is True

    # 0 = unlimited reads them all and is not flagged truncated.
    uncapped = ArchiveFileHandler(max_members=0).load(path)
    assert len(uncapped.members) == 10
    assert uncapped.truncated is False


def test_default_member_cap_is_applied() -> None:
    assert ArchiveFileHandler().max_members == DEFAULT_MAX_ARCHIVE_MEMBERS


def test_rejects_negative_caps() -> None:
    with pytest.raises(ValueError, match="max_members"):
        ArchiveFileHandler(max_members=-1)
    with pytest.raises(ValueError, match="max_member_content_bytes"):
        ArchiveFileHandler(max_member_content_bytes=-1)
    with pytest.raises(ValueError, match="max_entries"):
        ArchiveFileHandler(max_entries=-1)
    with pytest.raises(ValueError, match="max_total_content_bytes"):
        ArchiveFileHandler(max_total_content_bytes=-1)


def test_empty_archive_loads_without_error(tmp_path: Path) -> None:
    path = tmp_path / "empty.zip"
    with zipfile.ZipFile(path, "w"):
        pass

    handler = ArchiveFileHandler()
    payload = handler.load(path)
    assert len(payload.members) == 0
    # A memberless archive yields a degenerate (but valid) signal, never a crash.
    signal = handler.to_signal(payload)
    assert signal.size >= 1


def test_malformed_zip_load_raises_recoverably(tmp_path: Path) -> None:
    # A file claiming to be a zip but holding junk must raise (so the
    # Fingerprinter's per-handler try/except falls through to another handler),
    # never hang or corrupt -- mirroring the fail-soft routing philosophy.
    path = tmp_path / "bad.zip"
    path.write_bytes(b"PK\x03\x04 not a real zip at all " + b"x" * 200)

    with pytest.raises(Exception):  # noqa: B017,PT011 - any load failure is acceptable here
        ArchiveFileHandler().load(path)


def test_malformed_archive_falls_through_to_another_handler(tmp_path: Path) -> None:
    # End-to-end: a malformed archive does not abort fingerprinting; routing
    # degrades to a lower handler so the file is still fingerprinted.
    path = tmp_path / "bad.zip"
    path.write_bytes(b"PK\x03\x04 not a real zip " + b"x" * 600)

    fingerprint = Fingerprinter().fingerprint_file(path)
    assert fingerprint.handler != "archive"
    assert fingerprint.hash_count >= 0


def test_content_changed_same_name_size_still_differs(tmp_path: Path) -> None:
    # Two members with the SAME name and (near) same size but different bytes
    # must produce different identities via the small-member content digest, so
    # the archives are not falsely treated as identical.
    members_a = [("doc.txt", b"AAAA" * 100)]
    members_b = [("doc.txt", b"BBBB" * 100)]
    p_a = tmp_path / "a.zip"
    p_b = tmp_path / "b.zip"
    _make_zip(p_a, members_a)
    _make_zip(p_b, members_b)

    handler = ArchiveFileHandler()
    digest_a = handler.load(p_a).members[0].digest
    digest_b = handler.load(p_b).members[0].digest
    assert digest_a != digest_b


# ---------------------------------------------------------------------------
# F5: decompression-bomb bounds (aggregate budget + total-entry cap)
# ---------------------------------------------------------------------------


def test_default_new_caps_are_applied() -> None:
    handler = ArchiveFileHandler()
    assert handler.max_entries == DEFAULT_MAX_ARCHIVE_ENTRIES
    assert handler.max_total_content_bytes == DEFAULT_MAX_TOTAL_CONTENT_BYTES


def test_normal_archive_is_byte_identical_with_caps_default_vs_disabled(tmp_path: Path) -> None:
    # The new bounds must not perturb a normal small archive: members, sizes, and
    # content digests are identical with caps at their defaults vs fully disabled.
    path = tmp_path / "normal.zip"
    _make_zip(path, _base_members(12))

    default = ArchiveFileHandler().load(path)
    disabled = ArchiveFileHandler(
        max_entries=0, max_total_content_bytes=0
    ).load(path)

    as_ids = [(m.name, m.size, m.digest) for m in default.members]
    bs_ids = [(m.name, m.size, m.digest) for m in disabled.members]
    assert as_ids == bs_ids
    assert default.truncated is False
    # And the produced signal is identical too (same downstream hashes).
    handler = ArchiveFileHandler()
    assert (handler.to_signal(default) == handler.to_signal(disabled)).all()


def test_aggregate_content_budget_falls_back_to_identity_tokens(tmp_path: Path) -> None:
    # Many small members whose TOTAL content exceeds the aggregate budget: the
    # first members are content-digested (sha1, 40 hex chars), then the budget is
    # exhausted and the rest fall back to CRC tokens -- never a multi-GiB read.
    members = [(f"m{i}.bin", b"a" * (300 * 1024)) for i in range(6)]  # ~1.8 MiB total
    path = tmp_path / "big.zip"
    _make_zip(path, members)

    payload = ArchiveFileHandler(max_total_content_bytes=400 * 1024).load(path)
    sha1_members = [m for m in payload.members if len(m.digest) == 40]
    crc_members = [m for m in payload.members if m.digest.startswith("crc:")]
    assert sha1_members, "at least one member should be content-digested"
    assert crc_members, "the budget should force a fallback for later members"
    assert payload.truncated is True

    # With the budget disabled every member is content-digested (no fallback).
    uncapped = ArchiveFileHandler(max_total_content_bytes=0).load(path)
    assert all(len(m.digest) == 40 for m in uncapped.members)
    assert uncapped.truncated is False


def test_tar_total_entry_cap_counts_non_file_entries(tmp_path: Path) -> None:
    # The prior tar cap counted only regular files, so a flood of directory (or
    # symlink) entries iterated unbounded. The total-entry cap now stops it.
    path = tmp_path / "flood.tar"
    with tarfile.open(path, "w") as archive:
        for i in range(40):
            directory = tarfile.TarInfo(f"d{i}/")
            directory.type = tarfile.DIRTYPE
            archive.addfile(directory)
        data = b"hello world " * 50
        member = tarfile.TarInfo("real.txt")
        member.size = len(data)
        archive.addfile(member, io.BytesIO(data))

    capped = ArchiveFileHandler(max_entries=10).load(path)
    # Iteration stopped within the directory flood, before the trailing file.
    assert capped.truncated is True

    # With a generous entry cap the single real file is found and digested.
    full = ArchiveFileHandler(max_entries=1000).load(path)
    assert [m.name for m in full.members] == ["real.txt"]
    assert full.truncated is False


def test_aggregate_budget_never_raises_on_pathological_archive(tmp_path: Path) -> None:
    # Degrade-don't-abort: even when the aggregate budget trips, load() returns a
    # valid (truncated) payload rather than raising, preserving the fail-soft
    # contract. Members are content-eligible (<= per-member cap) but their TOTAL
    # exceeds the aggregate budget, so the budget -- not the per-member cap -- bites.
    members = [(f"m{i}.bin", b"z" * (50 * 1024)) for i in range(20)]
    path = tmp_path / "pathological.zip"
    _make_zip(path, members)
    payload = ArchiveFileHandler(
        max_total_content_bytes=128 * 1024, max_member_content_bytes=64 * 1024
    ).load(path)
    assert payload.members  # still produced a structural fingerprint
    assert payload.truncated is True


# ---------------------------------------------------------------------------
# Routing: .npz is a numpy vector container (a zip), owned by the embedding
# handler -- the generic archive handler must decline it despite the zip magic.
# ---------------------------------------------------------------------------


def test_npz_is_declined_by_archive_and_routes_to_embedding(tmp_path: Path) -> None:
    import numpy as np  # core dependency, always available

    npz = tmp_path / "vecs.npz"
    np.savez(npz, a=np.arange(96, dtype=np.float32).reshape(12, 8))
    sample = npz.read_bytes()[:16]
    # .npz carries the zip magic, so the magic sniff WOULD score it 0.90 and
    # out-rank the embedding handler's 0.80 -- the archive handler must decline.
    assert sample.startswith(b"PK")
    assert ArchiveFileHandler.can_handle(npz, sample=sample) == 0.0
    # End-to-end: a real .npz routes to the embedding handler, not archive, so it
    # gets the advertised vector-sequence fingerprint rather than a structural one.
    fingerprint = Fingerprinter().fingerprint_file(npz)
    assert fingerprint.handler == "embedding"


def test_real_zip_still_routes_to_archive(tmp_path: Path) -> None:
    # Regression guard: the .npz decline must not affect ordinary archives.
    path = tmp_path / "real.zip"
    _make_zip(path, _base_members(6))
    assert ArchiveFileHandler.can_handle(path, sample=path.read_bytes()[:16]) > 0.0
    assert Fingerprinter().fingerprint_file(path).handler == "archive"
