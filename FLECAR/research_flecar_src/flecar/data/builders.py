from __future__ import annotations

import random
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from flecar.data.contracts import validate_dataset
from flecar.utils import list_instance_files, read_json, write_json


@dataclass
class BuildConfig:
    source_instances: Path
    out_dir: Path
    target_instances: int = 360
    seed: int = 7


def _sample_xi(rng: random.Random) -> dict[str, Any]:
    dur_lo = round(rng.uniform(0.10, 0.25), 4)
    dur_hi = round(rng.uniform(max(dur_lo + 0.05, 0.20), 0.50), 4)
    return {
        "duration_interval_uncertainty": {
            "relative_range": [dur_lo, dur_hi],
            "distribution": "bounded_uniform",
        },
        "resource_occupancy_fluctuation": {
            "spike_probability": round(rng.uniform(0.05, 0.20), 4),
            "spike_magnitude": round(rng.uniform(0.10, 0.45), 4),
        },
        "temporary_agent_unavailability": {
            "event_probability": round(rng.uniform(0.03, 0.15), 4),
            "mean_window": int(rng.randint(1, 3)),
        },
        "task_failure_hazard_prior": {
            "base_hazard": round(rng.uniform(0.02, 0.10), 4),
            "critical_task_multiplier": round(rng.uniform(1.2, 2.0), 4),
        },
        "object_state_disturbance_prior": {
            "event_probability": round(rng.uniform(0.03, 0.12), 4),
            "state_reset_risk": round(rng.uniform(0.05, 0.20), 4),
        },
    }


def _inject_uncertainty_model(inst: dict[str, Any], rng: random.Random, clone_idx: int) -> None:
    xi = _sample_xi(rng)
    inst["Xi"] = xi
    inst.setdefault("metadata", {})["Xi"] = xi
    inst["metadata"]["xi_profile_id"] = f"xi_{clone_idx:04d}"


def _clone_instance(inst: dict[str, Any], clone_idx: int, rng: random.Random) -> dict[str, Any]:
    out = deepcopy(inst)
    out["episode_id"] = f"{inst['episode_id']}_fx{clone_idx:02d}"
    out["task_text"] = f"{inst['task_text']} [FLECAR synthetic-hard clone {clone_idx}]"

    cands = out["candidate_assignments"]
    by_subtask: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in cands:
        by_subtask[str(c["subtask_id"])].append(c)

    next_idx = max(int(str(c["candidate_id"]).lstrip("c")) for c in cands if str(c["candidate_id"]).lstrip("c").isdigit()) + 1

    for c in cands:
        if rng.random() < 0.70:
            c["start"] = max(0, int(c["start"]) + rng.choice([-1, 0, 1, 2, 3]))
        if rng.random() < 0.80:
            c["cost"] = round(float(c["cost"]) * (1.0 + rng.uniform(-0.08, 0.28)), 4)
        if rng.random() < 0.55:
            c["duration"] = max(1, int(c["duration"]) + rng.choice([0, 1, 2]))
        if rng.random() < 0.45 and c.get("resources"):
            res = list(c["resources"])
            res.append(f"synthetic_shared_{clone_idx % 7}")
            c["resources"] = sorted(set(str(x) for x in res))

    for sid, rows in list(by_subtask.items()):
        add_n = 1 + int(rng.random() < 0.35)
        for _ in range(add_n):
            if rng.random() < 0.65:
                base = rng.choice(rows)
                nc = deepcopy(base)
                nc["candidate_id"] = f"c{next_idx}"
                next_idx += 1
                nc["start"] = max(0, int(base["start"]) + rng.choice([1, 2, 3, 4]))
                nc["duration"] = max(1, int(base["duration"]) + rng.choice([0, 1, 2]))
                nc["cost"] = round(float(base["cost"]) * (1.0 + rng.uniform(0.05, 0.30)), 4)
                res = list(nc.get("resources", []))
                if rng.random() < 0.6:
                    res.append(f"synthetic_resource_{rng.randrange(12)}")
                nc["resources"] = sorted(set(str(x) for x in res))
                cands.append(nc)

    # Denser precedence/sync scaffolding for harder slices.
    if rng.random() < 0.40:
        subtasks = [str(s["id"]) for s in out.get("subtasks", [])]
        if len(subtasks) >= 3:
            u = rng.choice(subtasks[:-1])
            v = rng.choice(subtasks[1:])
            if u != v and [u, v] not in out.get("precedence", []):
                out.setdefault("precedence", []).append([u, v])

    if rng.random() < 0.35 and len(out.get("subtasks", [])) >= 2:
        sids = [str(s["id"]) for s in out["subtasks"]]
        a = rng.choice(sids)
        b = rng.choice([x for x in sids if x != a])
        out.setdefault("sync_constraints", []).append({"subtask_ids": [a, b], "max_lag": int(rng.choice([0, 1, 2]))})

    _inject_uncertainty_model(out, rng, clone_idx)

    out.setdefault("metadata", {})["synthetic_hard_clone"] = True
    out["metadata"]["clone_idx"] = clone_idx
    out["metadata"]["source_episode_id"] = str(inst["episode_id"])
    out["metadata"]["ecs_variant"] = "ECS-FCAR"
    return out


def build_flecar_dataset(cfg: BuildConfig) -> dict[str, Any]:
    rng = random.Random(cfg.seed)
    src_files = list_instance_files(cfg.source_instances)
    base_instances = [read_json(p) for p in src_files]
    if not base_instances:
        raise ValueError(f"no source instances found in {cfg.source_instances}")

    out_instances: list[dict[str, Any]] = []
    for i, inst in enumerate(base_instances):
        obj = deepcopy(inst)
        _inject_uncertainty_model(obj, rng, i)
        obj.setdefault("metadata", {})["ecs_variant"] = "ECS-FCAR"
        out_instances.append(obj)

    clone_idx = 0
    while len(out_instances) < cfg.target_instances:
        base = base_instances[clone_idx % len(base_instances)]
        out_instances.append(_clone_instance(base, clone_idx=clone_idx, rng=rng))
        clone_idx += 1

    out_instances = out_instances[: cfg.target_instances]
    validate_dataset(out_instances)

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    for i, inst in enumerate(out_instances):
        path = cfg.out_dir / f"instance_{i:04d}_{inst['episode_id']}.json"
        write_json(path, inst)

    summary = {
        "n_source_instances": len(base_instances),
        "n_output_instances": len(out_instances),
        "target_instances": cfg.target_instances,
        "seed": cfg.seed,
        "output_dir": str(cfg.out_dir),
        "ecs_problem_variant": "ECS-FCAR",
    }
    write_json(cfg.out_dir / "build_summary.json", summary)
    return summary
