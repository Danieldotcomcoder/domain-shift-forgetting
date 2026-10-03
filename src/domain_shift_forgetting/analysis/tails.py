"""Document-count trimming and signed token-contribution concentration."""

from dataclasses import dataclass
import math
import statistics
from typing import Sequence


@dataclass(frozen=True)
class DocumentEffect:
    document_id: str
    count: int
    d: float


@dataclass(frozen=True)
class TailSummary:
    d: float
    unweighted_median: float
    trimmed_token_weighted_mean: float
    trim_count_each_end: int
    signed_contributions: tuple[tuple[str, float], ...]
    top_signed_sum: float
    top_fraction_of_net: float | None
    top_fraction_of_positive_mass: float | None
    outlier_concentrated: bool


def summarize_documents(rows: Sequence[DocumentEffect], expected_d: float) -> TailSummary:
    if not rows or len({r.document_id for r in rows}) != len(rows):
        raise ValueError("Require nonempty unique documents")
    if not math.isfinite(expected_d) or any(not r.document_id or r.count <= 0 or not math.isfinite(r.d) for r in rows):
        raise ValueError("Invalid document statistics")
    total = sum(r.count for r in rows)
    d = math.fsum(r.count * r.d for r in rows) / total
    if abs(d - expected_d) > 1e-6:
        raise ValueError("Document contributions do not reconstruct full-dev D")
    ranked = sorted(rows, key=lambda r: (r.d, r.document_id))
    trim = math.floor(0.01 * len(rows))  # Explicit draft rounding convention.
    retained = ranked[trim:len(rows) - trim]
    trimmed = math.fsum(r.count * r.d for r in retained) / sum(r.count for r in retained)
    contributions = sorted(((r.document_id, r.count * r.d / total) for r in rows),
                           key=lambda item: (-item[1], item[0]))
    top = math.fsum(c for _, c in contributions[:math.ceil(0.01 * len(rows))])
    positive = math.fsum(max(c, 0.0) for _, c in contributions)
    ratio = top / d if d > 0 else None
    return TailSummary(d, statistics.median(r.d for r in rows), trimmed, trim,
                       tuple(contributions), top, ratio, top / positive if positive else None,
                       d >= 0.015 and top > 0.5 * d)
