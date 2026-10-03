"""Deterministic grouping helpers; draft hash conventions require freeze review."""

from dataclasses import dataclass
import hashlib
import json
from typing import Iterable


def content_sha256(text: str) -> str:
    """Draft canonicalization: original UTF-8 bytes, no whitespace/Unicode edits."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class RepositoryDocument:
    document_id: str
    aliases: tuple[str, ...]


def repository_groups(documents: Iterable[RepositoryDocument]) -> dict[str, str]:
    """Hash complete sorted alias components; must rebuild after pool expansion.

    Aliases are used exactly as supplied. Canonical repository identity extraction
    belongs to the dataset-specific adapter after T02 schema inspection.
    """
    rows = list(documents)
    if len({r.document_id for r in rows}) != len(rows):
        raise ValueError("Duplicate document ID")
    parents: dict[str, str] = {}

    def root(alias: str) -> str:
        parents.setdefault(alias, alias)
        while parents[alias] != alias:
            parents[alias] = parents[parents[alias]]
            alias = parents[alias]
        return alias

    for row in rows:
        if not row.document_id or not row.aliases or any(not a.strip() for a in row.aliases):
            raise ValueError("Drop missing-provenance rows before grouping")
        for alias in row.aliases:
            a, b = root(row.aliases[0]), root(alias)
            parents[max(a, b)] = min(a, b)
    components: dict[str, list[str]] = {}
    for alias in parents:
        components.setdefault(root(alias), []).append(alias)
    identities = {key: hashlib.sha256(json.dumps(sorted(aliases), ensure_ascii=False,
                  separators=(",", ":")).encode("utf-8")).hexdigest()
                  for key, aliases in components.items()}
    return {r.document_id: identities[root(r.aliases[0])] for r in rows}


def python_split(group_sha256: str) -> str:
    if len(group_sha256) != 64:
        raise ValueError("Expected SHA-256 group identity")
    bucket = int(group_sha256, 16) % 100
    return "train" if bucket < 90 else "dev" if bucket < 95 else "test"


def c4_validation_split(document_identity: str) -> str:
    """Draft convention: SHA-256 of UTF-8 identity, modulo two."""
    if not document_identity:
        raise ValueError("A stable identity is required")
    bucket = int(content_sha256(document_identity), 16) % 2
    return "dev" if bucket == 0 else "test"
