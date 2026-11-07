from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SolverConfig:
    name: str = "cpsat"
    time_budget_main_sec: float = 0.5
    time_budget_tight_sec: float = 0.25
    num_workers: int = 8


@dataclass
class CompilerConfig:
    # Canonical claim defaults.
    top_k: int = 2
    density_seed: int = 7
    constraint_density_scale: float = 1.5
    alpha_b: float = 0.25
    alpha_c: float = 0.3
    alpha_s: float = 0.15
    alpha_d: float = 0.1
    alpha_f: float = 0.2
    alpha_q: float = 0.2


@dataclass
class EnvelopeConfig:
    lambda_obj: float = 1.0
    lambda_flex: float = 0.4
    lambda_risk: float = 0.35
    slack_expand_ratio: float = 0.3
    min_window_pad: int = 1


@dataclass
class CARConfig:
    lambda_e: float = 4.0
    lambda_r: float = 1.0
    lambda_c: float = 2.0
    lambda_m: float = 1.5
    base_edit_penalty: float = 8.0
    localization_radius: int = 1


@dataclass
class ExperimentConfig:
    instances_path: Path = Path("data/instances/real_partnr/oracle_static")
    max_instances: int = 300
    seeds: list[int] = field(default_factory=lambda: [7, 11, 19, 23, 29])
    events_per_instance: int = 4


@dataclass
class FLECARConfig:
    solver: SolverConfig = field(default_factory=SolverConfig)
    compiler: CompilerConfig = field(default_factory=CompilerConfig)
    envelope: EnvelopeConfig = field(default_factory=EnvelopeConfig)
    car: CARConfig = field(default_factory=CARConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
