"""Fixed protocol clocks, explicit event coordinates, and planned run lineage."""

from dataclasses import dataclass
from enum import Enum
import math


class Condition(str, Enum):
    RMS = "RMS"
    TAPER_MINUS = "Taper-minus"
    TAPER_PLUS = "Taper-plus"


SEEDS = (101, 102, 103)
TOKENS_PER_UPDATE = 16_384
CALIBRATION_END = 763
GATE_END = 6_104
PREFIX_END = 9_156
CONTINUATION_UPDATES = 6_104
TRAJECTORY_END = 15_260
FULL_PREFIX = (763, 3052, 6104, 6409, 6714, 7019, 7324, 7629,
               7934, 8239, 8544, 8849, 9156)
QUICK_CONTINUATION = tuple(sorted(
    {0, 1, 2, 5, 10, 20, 1525, 3050, 6104}
    | set(range(50, 1001, 50)) | set(range(1100, 6101, 100))
))
FULL_CONTINUATION = tuple(sorted(
    {0, 1, 10, 100, 1525, 3050, 6104} | set(range(305, 6101, 305))
))
DIAGNOSTIC_CONTINUATION = (0, 10, 100, 1525, 6104)


def learning_rate(update: int) -> float:
    """The learning rate used by global optimizer update u (one-based)."""
    if not 1 <= update <= TRAJECTORY_END:
        raise ValueError("Optimizer update must be in 1..15260")
    if update <= 305:
        return 0.0006 * update / 305
    return 0.00006 + 0.5 * (0.0006 - 0.00006) * (
        1 + math.cos(math.pi * (update - 305) / (TRAJECTORY_END - 305))
    )


def gate(update: int) -> float:
    if not 1 <= update <= TRAJECTORY_END:
        raise ValueError("Optimizer update must be in 1..15260")
    if update <= CALIBRATION_END:
        return 1.0
    if update >= GATE_END:
        return 0.0
    return 0.5 * (1 + math.cos(math.pi * (update - CALIBRATION_END) / 5341))


@dataclass(frozen=True)
class EvaluationEvent:
    global_update: int
    continuation_update: int | None
    quick: bool
    full: bool
    diagnostics: bool


def prefix_events() -> tuple[EvaluationEvent, ...]:
    return tuple(EvaluationEvent(u, None, u == PREFIX_END, True, False)
                 for u in FULL_PREFIX)


def continuation_events() -> tuple[EvaluationEvent, ...]:
    points = set(QUICK_CONTINUATION) | set(FULL_CONTINUATION) | set(DIAGNOSTIC_CONTINUATION)
    return tuple(EvaluationEvent(PREFIX_END + s, s, s in QUICK_CONTINUATION,
                                 s in FULL_CONTINUATION, s in DIAGNOSTIC_CONTINUATION)
                 for s in sorted(points))


@dataclass(frozen=True)
class PlannedRun:
    run_id: str
    seed: int
    condition: Condition
    branch: str
    parent_run: str | None
    start_completed_update: int
    end_completed_update: int

    @property
    def planned_tokens(self) -> int:
        return (self.end_completed_update - self.start_completed_update) * TOKENS_PER_UPDATE


def planned_runs(include_auxiliary: bool) -> tuple[PlannedRun, ...]:
    """Expand an explicitly chosen allocation; never infer it from outcomes."""
    groups = [(s, c) for s in SEEDS for c in (Condition.RMS, Condition.TAPER_MINUS)]
    if include_auxiliary:
        groups.extend((s, Condition.TAPER_PLUS) for s in SEEDS)
    runs = []
    for seed, condition in groups:
        prefix_id = f"S{seed}-{condition.value}-prefix"
        runs.append(PlannedRun(prefix_id, seed, condition, "prefix", None, 0, PREFIX_END))
        for branch in ("web", "python"):
            runs.append(PlannedRun(f"S{seed}-{condition.value}-{branch}", seed, condition,
                                   branch, prefix_id, PREFIX_END, TRAJECTORY_END))
    return tuple(runs)
