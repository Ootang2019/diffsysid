from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def softplus_scalar(x: float) -> float:
    if x > 20.0:
        return x
    return math.log1p(math.exp(x))


def inv_softplus_scalar(y: float) -> float:
    if y > 20.0:
        return y
    return math.log(math.expm1(y))


def softplus_array(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    return np.where(x > 20.0, x, np.log1p(np.exp(x)))


def inv_softplus_array(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y)
    return np.where(y > 20.0, y, np.log(np.expm1(y)))


def resolve_sim_steps(sim_steps: int | None, trajectory_seconds: float, dt: float) -> int:
    if sim_steps is not None:
        return int(sim_steps)
    return max(1, int(round(float(trajectory_seconds) / float(dt))))


def sample_relative_init(
    rng: np.random.Generator,
    gt_value: float,
    mult_range: Sequence[float],
    min_gap_frac: float,
    name: str,
) -> float:
    lo, hi = (float(mult_range[0]), float(mult_range[1]))
    if lo <= 0.0 or hi <= 0.0 or lo > hi:
        raise ValueError(f"invalid {name} multiplier range: {(lo, hi)}")
    if not 0.0 <= min_gap_frac < 1.0:
        raise ValueError(f"{name} min gap fraction must be in [0, 1): {min_gap_frac}")

    for _ in range(256):
        mult = float(rng.uniform(lo, hi))
        if abs(mult - 1.0) >= min_gap_frac:
            return gt_value * mult

    raise RuntimeError(
        f"could not sample {name} multiplier with range {(lo, hi)} and min gap {min_gap_frac}; "
        "make the range wider or reduce the minimum gap"
    )


def sample_relative_init_vector(
    rng: np.random.Generator,
    gt: Sequence[float],
    mult_range: Sequence[float],
    min_gap_frac: float,
) -> list[float]:
    lo, hi = float(mult_range[0]), float(mult_range[1])
    gt_arr = np.asarray(gt, dtype=np.float64)
    for _ in range(256):
        mult = rng.uniform(lo, hi, size=len(gt_arr))
        if np.all(np.abs(mult - 1.0) >= min_gap_frac):
            return (gt_arr * mult).tolist()
    raise RuntimeError("could not sample sufficiently different initial parameters")
