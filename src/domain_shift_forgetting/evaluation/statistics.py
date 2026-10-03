"""Loss sums/counts preserve exact token-weighted aggregation."""

from dataclasses import dataclass, field
import math
from typing import Iterable


@dataclass
class LossSum:
    ce_sum: float = 0.0
    count: int = 0

    def add(self, value: float) -> None:
        if not math.isfinite(value) or value < 0:
            raise ValueError("CE must be finite and nonnegative")
        self.ce_sum += value
        self.count += 1

    @property
    def mean(self) -> float | None:
        return self.ce_sum / self.count if self.count else None


@dataclass
class EvaluationSums:
    total: LossSum = field(default_factory=LossSum)
    documents: dict[str, LossSum] = field(default_factory=dict)
    classes: dict[str, LossSum] = field(default_factory=lambda: {c: LossSum() for c in "WAPX"})
    rare: LossSum = field(default_factory=LossSum)

    def add(self, losses: Iterable[float], document_ids: Iterable[str],
            classes: Iterable[str], rare_flags: Iterable[bool]) -> None:
        for loss, doc, cls, rare in zip(losses, document_ids, classes, rare_flags, strict=True):
            if cls not in self.classes or not doc:
                raise ValueError("Invalid class or document identity")
            self.total.add(float(loss))
            self.documents.setdefault(doc, LossSum()).add(float(loss))
            self.classes[cls].add(float(loss))
            if rare:
                self.rare.add(float(loss))

    def code_ce(self, *, alphanumeric_punctuation_only: bool = False) -> float | None:
        included = "AP" if alphanumeric_punctuation_only else "APX"
        count = sum(self.classes[c].count for c in included)
        return sum(self.classes[c].ce_sum for c in included) / count if count else None

    def check_reconstruction(self, atol: float = 1e-6) -> None:
        for groups in (self.documents.values(), self.classes.values()):
            rows = list(groups)
            if sum(row.count for row in rows) != self.total.count:
                raise ValueError("Counts do not reconstruct corpus labels")
            if self.total.count and abs(sum(row.ce_sum for row in rows) - self.total.ce_sum) / self.total.count > atol:
                raise ValueError("CE sums do not reconstruct corpus CE")
