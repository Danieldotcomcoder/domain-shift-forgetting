"""Pure admission arithmetic. No timers, rentals, services, or watchdog loop."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class BudgetProjection:
    already_billed_seconds: float
    remaining_training_seconds: float
    remaining_evaluation_seconds: float
    remaining_other_seconds: float

    def __post_init__(self) -> None:
        if any(not math.isfinite(v) or v < 0 for v in (
            self.already_billed_seconds, self.remaining_training_seconds,
            self.remaining_evaluation_seconds, self.remaining_other_seconds,
        )):
            raise ValueError("Budget inputs must be measured/projected finite nonnegative seconds")

    @property
    def projected_total_with_margin(self) -> float:
        # Conservative convention: apply the specified margin to the total,
        # including already-billed work; no prior rental time is forgotten.
        return 1.15 * (self.already_billed_seconds + self.remaining_training_seconds
                       + self.remaining_evaluation_seconds + self.remaining_other_seconds)

    @property
    def fits(self) -> bool:
        return self.projected_total_with_margin <= 24 * 3600


def must_checkpoint_and_stop(*, total_billed_seconds: float, save_allowance_seconds: float) -> bool:
    if not all(math.isfinite(x) and x >= 0 for x in (total_billed_seconds, save_allowance_seconds)):
        raise ValueError("Invalid budget inputs")
    return total_billed_seconds + save_allowance_seconds >= 24 * 3600
