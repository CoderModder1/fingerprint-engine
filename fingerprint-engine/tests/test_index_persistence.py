"""Snapshot save/load durability: atomicity, .bak recovery, empty-save guard."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest
from _fixtures import (
    make_fingerprint,
)

from fingerprint_engine.core.exceptions import (
    SnapshotWriteRefused,
)
from fingerprint_engine.core.index import (
    InMemoryHashIndex,
)


def test_index_save_and_load_round_trips(tmp_path: Path) -> None:
    index_path = tmp_path / "index.json"
    index = InMemoryHashIndex()
    fingerprint = make_fingerprint("file-a", [1, 2, 3])
    index.add(fingerprint)
    index.save(index_path)

    loaded = InMemoryHashIndex.load(index_path)
    results = loaded.search(fingerprint)

    assert loaded.file_count == 1
    assert loaded.posting_count == 3
    assert results[0].file_id == "file-a"


def test_save_is_atomic_and_recovers_from_corrupt_primary(tmp_path: Path) -> None:
    # A durable save must leave no partial temp file in the directory, and a
    # corrupt primary with a valid .bak must transparently load from the backup.
    index_path = tmp_path / "index.json"
    index = InMemoryHashIndex()
    index.add(make_fingerprint("file-a", [1, 2, 3]))
    index.save(index_path)

    # No stray temp file left behind by the atomic write.
    assert [p.name for p in tmp_path.iterdir()] == ["index.json"]

    # Simulate a good backup next to a primary that was truncated mid-write.
    backup = index_path.with_name("index.json.bak")
    backup.write_text(index_path.read_text(encoding="utf-8"), encoding="utf-8")
    index_path.write_text('{"backend": "in_memory", "files": {"file-a": [[100', encoding="utf-8")

    loaded = InMemoryHashIndex.load(index_path)

    assert loaded.file_count == 1
    assert loaded.posting_count == 3


def test_save_keeps_backup_of_prior_contents(tmp_path: Path) -> None:
    # The second save must preserve the first save's snapshot at <dest>.bak so a
    # later corrupt primary can fall back to the previous good state.
    index_path = tmp_path / "index.json"
    backup = index_path.with_name("index.json.bak")

    first = InMemoryHashIndex()
    first.add(make_fingerprint("file-a", [1, 2, 3]))
    first.save(index_path)
    first_contents = index_path.read_text(encoding="utf-8")
    assert not backup.exists()  # nothing to back up on the first save

    second = InMemoryHashIndex()
    second.add(make_fingerprint("file-b", [4, 5]))
    second.save(index_path)

    # The .bak now holds the prior (file-a) snapshot, the primary the new one.
    assert backup.read_text(encoding="utf-8") == first_contents
    assert InMemoryHashIndex.load(backup).file_count == 1
    assert InMemoryHashIndex.load(index_path).file_count == 1
    assert InMemoryHashIndex.load(index_path).search(make_fingerprint("file-b", [4, 5]))[0].file_id == "file-b"


def test_load_raises_when_primary_corrupt_and_no_backup(tmp_path: Path) -> None:
    # A corrupt primary with no .bak must raise (not silently return an empty
    # index, which would then overwrite a good backup on the next save).
    index_path = tmp_path / "index.json"
    index_path.write_text('{"backend": "in_memory", "files": {"file-a"', encoding="utf-8")

    with pytest.raises(ValueError, match="corrupt"):
        InMemoryHashIndex.load(index_path)


def test_save_refuses_empty_over_nonempty_and_preserves_both_copies(tmp_path: Path) -> None:
    # A2: an empty (zero-file) save over a populated primary would clobber the
    # only good copy (the empty primary loads cleanly, so .bak never fires).
    # It must be refused, leaving BOTH the primary and any backup untouched.
    index_path = tmp_path / "index.json"

    populated = InMemoryHashIndex()
    populated.add(make_fingerprint("file-a", [1, 2, 3]))
    populated.save(index_path)
    populated.add(make_fingerprint("file-b", [4, 5]))
    populated.save(index_path)  # now a .bak (file-a) exists next to the primary
    primary_before = index_path.read_text(encoding="utf-8")
    backup = index_path.with_name("index.json.bak")
    backup_before = backup.read_text(encoding="utf-8")

    empty = InMemoryHashIndex()
    with pytest.raises(SnapshotWriteRefused) as excinfo:
        empty.save(index_path)
    assert excinfo.value.existing_file_count == 2

    # Neither the primary nor the backup was touched by the refused save.
    assert index_path.read_text(encoding="utf-8") == primary_before
    assert backup.read_text(encoding="utf-8") == backup_before
    assert InMemoryHashIndex.load(index_path).file_count == 2
    # No stray temp file leaked from the aborted write.
    assert not any(p.name.endswith(".tmp") for p in tmp_path.iterdir())


def test_save_empty_is_allowed_with_force_and_over_absent_or_empty(tmp_path: Path) -> None:
    # The guard fires ONLY for empty-over-non-empty. An empty save over an
    # absent or already-empty primary is fine, and force=True overrides the guard.
    empty = InMemoryHashIndex()
    fresh = tmp_path / "fresh.json"
    empty.save(fresh)  # over absent primary -> allowed
    assert InMemoryHashIndex.load(fresh).file_count == 0

    empty.save(fresh)  # over an already-empty primary -> allowed
    assert InMemoryHashIndex.load(fresh).file_count == 0

    populated = InMemoryHashIndex()
    populated.add(make_fingerprint("file-a", [1, 2, 3]))
    forced = tmp_path / "forced.json"
    populated.save(forced)
    InMemoryHashIndex().save(forced, force=True)  # explicit override -> allowed
    assert InMemoryHashIndex.load(forced).file_count == 0


def test_concurrent_saves_to_same_path_do_not_race_or_lose_writes(tmp_path: Path) -> None:
    # A3: two threads saving the same destination in one process previously
    # shared a PID-only temp name and raced (FileNotFoundError / torn write).
    # With a unique-per-writer temp, every save completes and the file is always
    # a valid, fully-written snapshot.
    index_path = tmp_path / "index.json"
    index = InMemoryHashIndex()
    index.add(make_fingerprint("file-a", [1, 2, 3]))
    index.save(index_path)  # seed so all saves are non-empty (guard never fires)

    errors: list[Exception] = []
    barrier = threading.Barrier(8)

    def _saver() -> None:
        barrier.wait()
        try:
            for _ in range(12):
                index.save(index_path)
        except Exception as exc:  # noqa: BLE001 - capture any race for the assertion
            errors.append(exc)

    threads = [threading.Thread(target=_saver) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    # The destination is always a complete, loadable snapshot and no temp leaked.
    assert InMemoryHashIndex.load(index_path).file_count == 1
    assert not any(p.name.endswith(".tmp") for p in tmp_path.iterdir())

