"""Duplicate detection over a set of input files.

Two tiers, cheapest first:

* **Exact** -- group the input paths by ``content_sha256`` (which equals a
  fingerprint's ``file_id``). Any sha shared by more than one input path is an
  exact-duplicate cluster. This is byte-level identity, needs no fuzzy
  comparison, and short-circuits those paths out of the fuzzy tier entirely.
* **Near-duplicate** -- index one representative per *distinct* content, then run
  the existing :meth:`HashIndex.search` for each representative and connect any
  two distinct-content files whose pairwise match ``confidence`` (the existing
  handler-comparable :attr:`SearchResult.confidence`) is at least
  ``min_confidence``. Connected files are unioned into a near-duplicate cluster
  (transitive via union-find), so a chain A~B~C lands in one cluster.

The engine de-dupes the index by content (``file_id == content_sha256``), so two
byte-identical inputs collapse to a single index entry -- which is exactly why the
exact tier must work from the *input paths*, not from the index. Each input file
is fingerprinted at most once (the caller passes the fingerprints in), so the
cost is ``O(n * search)`` rather than ``O(n^2)`` re-fingerprinting.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from .index import HashIndex, InMemoryHashIndex
from .models import Fingerprint

# Default pairwise confidence cutoff for the near-duplicate tier. Matches the
# search confidence scale: 1.0 is a self/identical match, 0.0 is no alignment.
DEFAULT_MIN_CONFIDENCE = 0.5


@dataclass(frozen=True)
class ExactDuplicateCluster:
    """A group of input paths whose bytes are identical (same ``content_sha256``)."""

    content_sha256: str
    paths: list[str]

    def to_dict(self) -> dict[str, object]:
        return {"content_sha256": self.content_sha256, "paths": list(self.paths)}


@dataclass(frozen=True)
class NearDuplicateCluster:
    """A group of distinct-content input files linked by fingerprint similarity.

    ``paths`` holds one path per distinct content in the cluster (the
    representative path for each ``content_sha256``). ``confidence`` is the
    strongest pairwise match confidence observed among the cluster's members --
    a single comparable number for "how near" these near-duplicates are.
    """

    paths: list[str]
    confidence: float
    file_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "paths": list(self.paths),
            "file_ids": list(self.file_ids),
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class DedupReport:
    """Structured outcome of :func:`find_duplicates`.

    * ``exact`` -- byte-identical clusters (each two or more input paths).
    * ``near`` -- fuzzy-similar clusters across *distinct* content.
    * ``singletons`` -- distinct contents that landed in no cluster of either
      tier (unique, no near-duplicate). Counts distinct content, not paths.
    * ``total_paths`` / ``total_distinct`` -- input accounting.
    """

    exact: list[ExactDuplicateCluster]
    near: list[NearDuplicateCluster]
    singletons: int
    total_paths: int
    total_distinct: int

    def to_dict(self) -> dict[str, object]:
        return {
            "exact_clusters": [cluster.to_dict() for cluster in self.exact],
            "near_duplicate_clusters": [cluster.to_dict() for cluster in self.near],
            "singletons": self.singletons,
            "total_paths": self.total_paths,
            "total_distinct": self.total_distinct,
            "exact_cluster_count": len(self.exact),
            "near_duplicate_cluster_count": len(self.near),
        }


class _UnionFind:
    """Minimal union-find over hashable keys for transitive clustering."""

    def __init__(self) -> None:
        self._parent: dict[str, str] = {}

    def add(self, key: str) -> None:
        self._parent.setdefault(key, key)

    def find(self, key: str) -> str:
        self.add(key)
        root = key
        while self._parent[root] != root:
            root = self._parent[root]
        # Path compression keeps repeated finds near-constant.
        while self._parent[key] != root:
            self._parent[key], key = root, self._parent[key]
        return root

    def union(self, left: str, right: str) -> None:
        self._parent[self.find(right)] = self.find(left)

    def groups(self) -> dict[str, list[str]]:
        clusters: dict[str, list[str]] = {}
        for key in self._parent:
            clusters.setdefault(self.find(key), []).append(key)
        return clusters


def find_duplicates(
    fingerprints: Iterable[Fingerprint],
    *,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    index: HashIndex | None = None,
) -> DedupReport:
    """Cluster ``fingerprints`` into exact and near-duplicate groups.

    Each :class:`Fingerprint` is the result of fingerprinting one input path; its
    ``file_id`` equals its ``content_sha256`` and its ``path`` is the input file
    (so we report paths, not opaque ids). Fingerprints whose ``path`` is empty
    fall back to their ``file_id`` for reporting.

    Tier 1 groups by ``content_sha256``; any sha with two or more *paths* is an
    exact cluster and those paths are excluded from the fuzzy tier (one
    representative per distinct content carries forward). Tier 2 indexes one
    representative per distinct content into ``index`` (a fresh
    :class:`InMemoryHashIndex` if none is supplied -- callers may pass another
    backend), searches each representative, and unions any two *distinct*-content
    representatives whose match confidence is at least ``min_confidence`` into a
    near-duplicate cluster.

    The supplied/created index is treated as scratch space for this call; the
    representatives are added to it. Pass a fresh index (the default) unless you
    intend to dedup against an already-populated one.
    """

    # --- Materialize inputs, keeping first-seen order for deterministic output.
    by_sha: dict[str, list[str]] = {}
    representative: dict[str, Fingerprint] = {}
    sha_order: list[str] = []
    total_paths = 0
    for fingerprint in fingerprints:
        total_paths += 1
        sha = fingerprint.content_sha256
        path = fingerprint.path or fingerprint.file_id
        if sha not in by_sha:
            by_sha[sha] = []
            representative[sha] = fingerprint
            sha_order.append(sha)
        by_sha[sha].append(path)

    # --- TIER 1: exact (byte-identical) clusters, in first-seen order.
    exact: list[ExactDuplicateCluster] = [
        ExactDuplicateCluster(content_sha256=sha, paths=list(by_sha[sha]))
        for sha in sha_order
        if len(by_sha[sha]) > 1
    ]

    # --- TIER 2: near-duplicate clustering over one representative per content.
    near = _near_duplicate_clusters(
        sha_order=sha_order,
        by_sha=by_sha,
        representative=representative,
        min_confidence=min_confidence,
        index=index if index is not None else InMemoryHashIndex(),
    )

    # Distinct contents that ended up in no cluster of either tier.
    clustered_sha = {sha for sha in sha_order if len(by_sha[sha]) > 1}
    for cluster in near:
        clustered_sha.update(cluster.file_ids)
    singletons = sum(1 for sha in sha_order if sha not in clustered_sha)

    return DedupReport(
        exact=exact,
        near=near,
        singletons=singletons,
        total_paths=total_paths,
        total_distinct=len(sha_order),
    )


def _near_duplicate_clusters(
    *,
    sha_order: Sequence[str],
    by_sha: Mapping[str, list[str]],
    representative: Mapping[str, Fingerprint],
    min_confidence: float,
    index: HashIndex,
) -> list[NearDuplicateCluster]:
    """Union representatives whose pairwise search confidence clears the cutoff.

    Indexes one representative per distinct content, searches each one, and joins
    any two distinct representatives meeting ``min_confidence``. Self-matches (a
    representative matching its own ``file_id``) are ignored. Returns clusters
    that contain two or more *distinct* contents; a lone representative with no
    qualifying neighbor is a singleton, not a cluster.
    """

    # Index one representative per distinct content. add_many is last-wins per
    # file_id and file_id == content_sha256, so the distinct reps survive 1:1.
    index.add_many(representative[sha] for sha in sha_order)

    union = _UnionFind()
    for sha in sha_order:
        union.add(sha)

    # Strongest confidence seen on any edge touching each representative, so a
    # cluster can report a single comparable "how near" number.
    best_confidence: dict[str, float] = {sha: 0.0 for sha in sha_order}
    for sha in sha_order:
        query = representative[sha]
        for result in index.search(query, top_k=len(sha_order) + 1):
            other = result.file_id
            if other == sha or other not in best_confidence:
                continue  # self-match or a stray id not in this batch
            if result.confidence >= min_confidence:
                union.union(sha, other)
                best_confidence[sha] = max(best_confidence[sha], result.confidence)
                best_confidence[other] = max(best_confidence[other], result.confidence)

    # Build clusters of size >= 2 (distinct contents), each in first-seen order
    # by representative, and the cluster list itself ordered by its first member.
    rank = {sha: position for position, sha in enumerate(sha_order)}
    clusters: list[NearDuplicateCluster] = []
    for members in union.groups().values():
        if len(members) < 2:
            continue
        members_sorted = sorted(members, key=lambda sha: rank[sha])
        clusters.append(
            NearDuplicateCluster(
                paths=[by_sha[sha][0] for sha in members_sorted],
                file_ids=list(members_sorted),
                confidence=round(max(best_confidence[sha] for sha in members_sorted), 6),
            )
        )
    clusters.sort(key=lambda cluster: rank[cluster.file_ids[0]])
    return clusters
