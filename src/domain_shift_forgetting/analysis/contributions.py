"""Reconstruct paired effects from identical full-development label populations."""

from dataclasses import dataclass

from domain_shift_forgetting.evaluation.statistics import EvaluationSums, LossSum
from .tails import DocumentEffect


@dataclass(frozen=True)
class ClassEffect:
    token_class: str
    count: int
    mean_d: float | None
    weighted_contribution: float


def _four_way(tp: LossSum, tw: LossSum, rp: LossSum, rw: LossSum) -> float | None:
    if len({r.count for r in (tp, tw, rp, rw)}) != 1:
        raise ValueError("Different scored-label populations across conditions/branches")
    if not tp.count:
        return None
    return ((tp.ce_sum - tw.ce_sum) - (rp.ce_sum - rw.ce_sum)) / tp.count


def decompose(taper_python: EvaluationSums, taper_web: EvaluationSums,
              rms_python: EvaluationSums, rms_web: EvaluationSums
              ) -> tuple[float, tuple[DocumentEffect, ...], tuple[ClassEffect, ...]]:
    """Caller must bind all four statistics to the same full-dev array hash/event."""
    groups = (taper_python, taper_web, rms_python, rms_web)
    for group in groups:
        group.check_reconstruction()
    if any(set(g.documents) != set(taper_python.documents) for g in groups):
        raise ValueError("Different document populations across branches")
    d = _four_way(*(g.total for g in groups))
    if d is None:
        raise ValueError("Empty full-development evaluation")
    documents = []
    for doc in sorted(taper_python.documents):
        effect = _four_way(*(g.documents[doc] for g in groups))
        if effect is None:
            raise ValueError("Unscored document in sufficient statistics")
        documents.append(DocumentEffect(doc, taper_python.documents[doc].count, effect))
    classes = []
    for cls in "WAPX":
        effect = _four_way(*(g.classes[cls] for g in groups))
        count = taper_python.classes[cls].count
        classes.append(ClassEffect(cls, count, effect, (effect or 0.0) * count / taper_python.total.count))
    if abs(sum(c.weighted_contribution for c in classes) - d) > 1e-6:
        raise ValueError("Disjoint classes do not reconstruct D")
    return d, tuple(documents), tuple(classes)
