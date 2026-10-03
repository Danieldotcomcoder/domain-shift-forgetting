"""Exact duplicate precedence and exact verification of near-duplicate candidates.

MinHash/LSH candidate retrieval and deterministic replenishment are intentionally
not implemented until preparation dependencies and actual schema are pinned.
"""

from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class DocumentIdentity:
    document_id: str
    content_sha256: str
    domain: str
    split: str


@dataclass(frozen=True)
class Removal:
    removed_id: str
    retained_id: str
    reason: str


def exact_dedup(rows: Iterable[DocumentIdentity]) -> tuple[tuple[DocumentIdentity, ...], tuple[Removal, ...]]:
    """Across both domains: test wins over dev, which wins over train.

    Draft same-split tie convention: lexicographic document ID. Replenishment
    must occur outside this function, followed by grouping and another audit.
    """
    priority = {"test": 0, "dev": 1, "train": 2}
    documents = list(rows)
    if len({r.document_id for r in documents}) != len(documents):
        raise ValueError("Document IDs must be globally unique")
    for row in documents:
        if not row.document_id or row.split not in priority or row.domain not in {"web", "python"}:
            raise ValueError("Invalid document identity")
        if len(row.content_sha256) != 64 or any(c not in "0123456789abcdef" for c in row.content_sha256):
            raise ValueError("Invalid content hash")
    winners: dict[str, DocumentIdentity] = {}
    removals = []
    for row in sorted(documents, key=lambda r: (priority[r.split], r.document_id)):
        retained = winners.get(row.content_sha256)
        if retained is None:
            winners[row.content_sha256] = row
        else:
            removals.append(Removal(row.document_id, retained.document_id, "exact_content_duplicate"))
    return tuple(sorted(winners.values(), key=lambda r: r.document_id)), tuple(removals)


def token_fivegrams(tokens: Sequence[int]) -> frozenset[tuple[int, ...]]:
    return frozenset(tuple(tokens[i:i + 5]) for i in range(max(0, len(tokens) - 4)))


def candidate_jaccard(left: Sequence[int], right: Sequence[int]) -> float | None:
    a, b = token_fivegrams(left), token_fivegrams(right)
    union = a | b
    # Short documents cannot be audited with this representation; report them,
    # rather than pretending empty shingles establish a near-duplicate match.
    return len(a & b) / len(union) if a and b else None
