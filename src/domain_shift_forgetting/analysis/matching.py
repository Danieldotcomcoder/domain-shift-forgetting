"""First observed downward-crossing matching; no smoothing or extrapolation."""

from dataclasses import dataclass
import math
from typing import Sequence


@dataclass(frozen=True)
class AdaptationPoint:
    update: int
    non_w_code_ce: float
    web_ce: float

    def __post_init__(self) -> None:
        if not 0 <= self.update <= 6104 or any(not math.isfinite(v) or v < 0 for v in (self.non_w_code_ce, self.web_ce)):
            raise ValueError("Invalid adaptation observation")


@dataclass(frozen=True)
class Match:
    target_code_ce: float
    left: AdaptationPoint
    right: AdaptationPoint
    fraction: float
    update: float
    web_ce: float
    nearest_endpoint_web_ce: float

    @property
    def tokens(self) -> float:
        return self.update * 16384

    @property
    def bracket_updates(self) -> int:
        return self.right.update - self.left.update


def first_crossing(points: Sequence[AdaptationPoint], target: float) -> Match | None:
    if not points or points[0].update != 0:
        raise ValueError("Matching needs the s=0 baseline")
    if not math.isfinite(target) or target < 0:
        raise ValueError("Invalid matching target")
    if any(a.update >= b.update for a, b in zip(points, points[1:])):
        raise ValueError("Observations must be strictly chronological")
    if points[0].non_w_code_ce < target:
        return None  # Already surpassed; do not manufacture a post-switch match.
    for index, point in enumerate(points):
        if point.non_w_code_ce == target:
            return Match(target, point, point, 0.0, float(point.update), point.web_ce, point.web_ce)
        if index == 0:
            continue
        left = points[index - 1]
        if left.non_w_code_ce > target > point.non_w_code_ce:
            # The first downward crossing is ineligible if its observed gap is
            # too wide; never select a later recrossing instead.
            if point.update - left.update > 100:
                return None
            fraction = (left.non_w_code_ce - target) / (left.non_w_code_ce - point.non_w_code_ce)
            web = left.web_ce + fraction * (point.web_ce - left.web_ce)
            update = left.update + fraction * (point.update - left.update)
            nearest = left if fraction <= 0.5 else point
            return Match(target, left, point, fraction, update, web, nearest.web_ce)
    return None


def matched_forgetting(rms: Sequence[AdaptationPoint], taper: Sequence[AdaptationPoint],
                       target_update: int) -> tuple[Match | None, float | None]:
    if target_update not in (1525, 3050, 6104):
        raise ValueError("Target must be a frozen RMS checkpoint")
    if not rms or rms[0].update != 0 or not taper or taper[0].update != 0:
        raise ValueError("Both conditions need s=0")
    if any(a.update >= b.update for a, b in zip(rms, rms[1:])):
        raise ValueError("RMS observations must be strictly chronological")
    targets = [p for p in rms if p.update == target_update]
    if len(targets) != 1:
        raise ValueError("Missing or duplicate RMS target observation")
    target = targets[0]
    match = first_crossing(taper, target.non_w_code_ce)
    if match is None:
        return None, None
    difference = (match.web_ce - taper[0].web_ce) - (target.web_ce - rms[0].web_ce)
    return match, difference
