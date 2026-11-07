from __future__ import annotations

import hashlib
import random
from typing import Any

from qcma.repair.event_simulator import EVENT_TYPES, apply_event_to_bans, edit_distance_by_subtask
from qcma.solvers.common import is_feasible

from flecar.repair.normalization import clip01


def deterministic_rng(*parts: Any) -> random.Random:
    key = "||".join(str(p) for p in parts)
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    seed = int.from_bytes(digest, byteorder="big", signed=False)
    return random.Random(seed)


def common_next_event_survivability(
    reduced: dict[str, Any],
    selected: list[str],
    *,
    trigger_time: int,
    rng: random.Random,
) -> int:
    event_type = EVENT_TYPES[rng.randrange(len(EVENT_TYPES))]
    bans2 = apply_event_to_bans(
        reduced,
        selected,
        trigger_time=int(trigger_time) + 1,
        event_type=str(event_type),
        rng=rng,
    )
    kept = [cid for cid in selected if str(cid) not in {str(x) for x in bans2}]
    return int(is_feasible(reduced, kept))


def normalized_repair_burden(
    reduced: dict[str, Any],
    prev_selected: list[str],
    new_selected: list[str],
    *,
    feasible: bool,
    repair_latency: float,
    repair_triggered: int,
    time_budget_sec: float,
) -> dict[str, float]:
    n_subtasks = max(1, len(reduced.get("unique_groups", {})))
    edits = int(edit_distance_by_subtask(reduced, prev_selected, new_selected))
    edit_frac = clip01(float(edits) / float(n_subtasks))
    latency_frac = clip01(float(repair_latency) / max(1e-9, float(time_budget_sec)))
    trigger_frac = 1.0 if int(repair_triggered) else 0.0
    failure_frac = 0.0 if bool(feasible) else 1.0
    burden = float((edit_frac + latency_frac + trigger_frac + failure_frac) / 4.0)
    return {
        "repair_burden": burden,
        "edit_burden": float(edits),
        "edit_fraction": float(edit_frac),
        "latency_fraction": float(latency_frac),
        "trigger_fraction": float(trigger_frac),
        "failure_fraction": float(failure_frac),
    }
