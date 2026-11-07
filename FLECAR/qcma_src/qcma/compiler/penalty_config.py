"""Penalty configuration for QUBO compilation."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class PenaltyConfig:
    p_assign: float = 100.0
    p_conflict: float = 120.0
    p_precedence: float = 120.0
    p_sync: float = 100.0
    lambda_cost: float = 1.0
    lambda_repair: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {k: float(v) for k, v in asdict(self).items()}

