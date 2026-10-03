"""Ordered pilot screening rules. This module never initiates another study."""

from dataclasses import dataclass, field
import math
import statistics
from typing import Mapping, Sequence

from domain_shift_forgetting.protocol import QUICK_CONTINUATION, SEEDS


@dataclass(frozen=True)
class SeedEvidence:
    seed: int
    d: float
    d3050: float
    q: float
    absolute_relative_prefix_gap: float
    rms_non_w_improvement: float
    taper_non_w_improvement: float
    quick_d: Mapping[int, float]
    matched_differences: Mapping[int, float | None]


@dataclass(frozen=True)
class Decision:
    category: str
    reasons: tuple[str, ...]
    common_target_update: int | None = None
    statistics: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class DecisionThresholds:
    """Freeze BEFORE training; hours do not replace matched scientific endpoints.

    These defaults transcribe protocol v3. A change requires a new protocol ID,
    never tuning after inspecting seed outcomes. Units are CE nats/token except
    prefix_gap (a fraction relative to the RMS prefix) and q_fraction.
    """
    prefix_gap: float = 0.02
    adaptation: float = 0.05
    proceed_d: float = 0.03
    proceed_d3050: float = 0.015
    q_fraction: float = 0.5
    opposite_d: float = -0.03
    small_d: float = 0.015
    transient_d: float = 0.05

    def __post_init__(self):
        if not all(math.isfinite(v) for v in self.__dict__.values()):
            raise ValueError("Decision thresholds must be finite")
        if not (0 <= self.prefix_gap and 0 <= self.adaptation and
                self.opposite_d < 0 < self.small_d <= self.proceed_d and
                0 <= self.proceed_d3050 <= self.proceed_d and
                0 < self.q_fraction <= 1 and self.transient_d > 0):
            raise ValueError("Invalid decision threshold ordering")


def classify(seeds: Sequence[SeedEvidence], *,
             primary_complete: bool, correctness_and_data_passed: bool,
             nonfinite_primary: bool = False, resource_terminated: bool = False,
             thresholds: DecisionThresholds = DecisionThresholds()) -> Decision:
    invalid = []
    if not primary_complete or len(seeds) != 3 or {s.seed for s in seeds} != set(SEEDS):
        invalid.append("Missing or duplicate planned primary seed")
    if not correctness_and_data_passed:
        invalid.append("Unresolved correctness/data gate")
    if nonfinite_primary or resource_terminated:
        invalid.append("Nonfinite primary run or resource termination")
    for seed in seeds:
        required = (seed.d, seed.d3050, seed.q, seed.absolute_relative_prefix_gap,
                    seed.rms_non_w_improvement, seed.taper_non_w_improvement)
        if not all(math.isfinite(v) for v in required) or seed.absolute_relative_prefix_gap < 0:
            invalid.append(f"Invalid primary evidence for seed {seed.seed}")
        if set(seed.quick_d) != set(QUICK_CONTINUATION) or not all(math.isfinite(v) for v in seed.quick_d.values()):
            invalid.append(f"Missing/nonfinite quick-dev evidence for seed {seed.seed}")
        if set(seed.matched_differences) != {1525, 3050, 6104} or any(
            v is not None and not math.isfinite(v) for v in seed.matched_differences.values()
        ):
            invalid.append(f"Missing/nonfinite matching records for seed {seed.seed}")
    if invalid:
        return Decision("INVALID OR INCOMPLETE", tuple(invalid))
    limited = [f"Primary guardrail failed for seed {s.seed}" for s in seeds
               if s.absolute_relative_prefix_gap > thresholds.prefix_gap or s.rms_non_w_improvement < thresholds.adaptation
               or s.taper_non_w_improvement < thresholds.adaptation]
    if limited:
        return Decision("COMPARABILITY / ADAPTATION LIMITED", tuple(limited))
    mean_d = statistics.mean(s.d for s in seeds)
    stats = {"mean_D": mean_d, "mean_D3050": statistics.mean(s.d3050 for s in seeds),
             "mean_Q": statistics.mean(s.q for s in seeds)}
    common = [u for u in (1525, 3050, 6104) if all(s.matched_differences[u] is not None for s in seeds)]
    target = max(common) if common else None
    if target is not None:
        stats["mean_matched_difference"] = statistics.mean(float(s.matched_differences[target]) for s in seeds)
    if mean_d <= thresholds.opposite_d:
        return Decision("OPPOSITE DIRECTION", (f"Mean D <= {thresholds.opposite_d}",), target, stats)
    if (mean_d >= thresholds.proceed_d and all(s.d > 0 for s in seeds) and stats["mean_D3050"] >= thresholds.proceed_d3050
            and stats["mean_Q"] > thresholds.q_fraction * mean_d and target is not None
            and stats["mean_matched_difference"] > 0):
        return Decision("PROCEED TO DESIGN THE NEXT STUDY", ("All pilot screening criteria met; no confirmatory claim",), target, stats)
    if mean_d < thresholds.small_d:
        transient = any(statistics.mean(s.quick_d[u] for s in seeds) >= thresholds.transient_d
                        for u in QUICK_CONTINUATION if u <= 1000)
        category = "TRANSIENT ONLY" if transient else "STOP — SMALL OBSERVED EFFECT"
        return Decision(category, (f"Mean endpoint D < {thresholds.small_d}; not evidence of equivalence",), target, stats)
    return Decision("INCONCLUSIVE", ("Screening criteria not all met",), target, stats)
