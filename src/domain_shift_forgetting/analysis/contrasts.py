"""Token-weighted web-CE estimands and descriptive seed uncertainty."""

from dataclasses import dataclass
import math
import statistics
from typing import Sequence


@dataclass(frozen=True)
class BranchCE:
    rms_prefix: float
    taper_prefix: float
    rms_web: float
    rms_python: float
    taper_web: float
    taper_python: float

    def __post_init__(self) -> None:
        if any(not math.isfinite(x) or x < 0 for x in self.__dict__.values()):
            raise ValueError("All branch values must be finite nonnegative web CE")


@dataclass(frozen=True)
class Contrast:
    d: float
    g_web: float
    q: float
    f_rms_web: float
    f_rms_python: float
    f_taper_web: float
    f_taper_python: float


def contrast(ce: BranchCE) -> Contrast:
    d = (ce.taper_python - ce.taper_web) - (ce.rms_python - ce.rms_web)
    g = (ce.taper_prefix - ce.rms_prefix) - (ce.taper_web - ce.rms_web)
    frw, frp = ce.rms_web - ce.rms_prefix, ce.rms_python - ce.rms_prefix
    ftw, ftp = ce.taper_web - ce.taper_prefix, ce.taper_python - ce.taper_prefix
    q = d - g
    if not math.isclose(q, ftp - frp, rel_tol=0, abs_tol=1e-6):
        raise ValueError("Forgetting decomposition does not reconstruct")
    return Contrast(d, g, q, frw, frp, ftw, ftp)


@dataclass(frozen=True)
class SeedUncertainty:
    values: tuple[float, ...]
    mean: float
    sample_sd: float
    descriptive_t95: tuple[float, float]


def seed_uncertainty(values: Sequence[float]) -> SeedUncertainty:
    if len(values) != 3 or not all(math.isfinite(v) for v in values):
        raise ValueError("The descriptive interval requires all three finite paired seeds")
    mean, sd = statistics.mean(values), statistics.stdev(values)
    halfwidth = 4.303 * sd / math.sqrt(3)
    return SeedUncertainty(tuple(values), mean, sd, (mean - halfwidth, mean + halfwidth))


def prefix_slope(points: Sequence[tuple[int, float]]) -> float:
    """OLS CE slope in nats per million supervised tokens, last five observations."""
    if len(points) < 5 or any(points[i][0] >= points[i + 1][0] for i in range(len(points) - 1)):
        raise ValueError("Need at least five chronological prefix points")
    tail = points[-5:]
    if any(not math.isfinite(y) for _, y in tail):
        raise ValueError("Nonfinite prefix loss")
    xs = [u * 16384 / 1e6 for u, _ in tail]
    ys = [y for _, y in tail]
    xm, ym = statistics.mean(xs), statistics.mean(ys)
    return sum((x - xm) * (y - ym) for x, y in zip(xs, ys)) / sum((x - xm) ** 2 for x in xs)
