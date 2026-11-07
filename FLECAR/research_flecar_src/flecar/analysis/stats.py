from __future__ import annotations

import math
import random
from typing import Any


def mean(vals: list[float]) -> float:
    return float(sum(vals) / max(1, len(vals)))


def bootstrap_ci(vals: list[float], *, n_boot: int = 2000, alpha: float = 0.05, seed: int = 7) -> tuple[float, float]:
    if not vals:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(vals)
    boots: list[float] = []
    for _ in range(n_boot):
        sample = [vals[rng.randrange(n)] for _ in range(n)]
        boots.append(mean(sample))
    boots.sort()
    lo_idx = max(0, int((alpha / 2.0) * len(boots)) - 1)
    hi_idx = min(len(boots) - 1, int((1.0 - alpha / 2.0) * len(boots)) - 1)
    return float(boots[lo_idx]), float(boots[hi_idx])


def exact_sign_test_p_two_sided(wins: int, losses: int) -> float:
    n = wins + losses
    if n <= 0:
        return 1.0
    k = min(wins, losses)
    p = 0.0
    for i in range(0, k + 1):
        p += math.comb(n, i)
    p = min(1.0, 2.0 * p / (2 ** n))
    return float(p)


def paired_summary(
    rows: list[dict[str, Any]],
    *,
    sched_key: str,
    base_key: str,
    metric_key: str,
    seed: int = 7,
) -> dict[str, Any]:
    deltas = [float(r[sched_key]) - float(r[base_key]) for r in rows]
    wins = sum(1 for d in deltas if d > 1e-12)
    losses = sum(1 for d in deltas if d < -1e-12)
    ties = len(deltas) - wins - losses
    lo, hi = bootstrap_ci(deltas, seed=seed)
    return {
        "n_pairs": len(deltas),
        "metric": metric_key,
        "delta_mean": mean(deltas),
        "delta_ci95_low": lo,
        "delta_ci95_high": hi,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "p_two_sided": exact_sign_test_p_two_sided(wins, losses),
    }
