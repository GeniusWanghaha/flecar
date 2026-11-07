"""CP-SAT based repair solver with frozen prefix and edit penalty."""

from __future__ import annotations

from time import perf_counter
from typing import Any

from ortools.sat.python import cp_model

from qcma.solvers.common import is_feasible, objective_from_selection


def solve_repair_cpsat(
    reduced: dict[str, Any],
    prev_selected: list[str],
    *,
    frozen_subtasks: set[str] | None = None,
    banned_candidates: set[str] | None = None,
    editable_subtasks: set[str] | None = None,
    edit_penalty: float = 0.0,
    time_limit_sec: float = 0.5,
    num_workers: int = 8,
) -> dict[str, Any]:
    if frozen_subtasks is None:
        frozen_subtasks = set()
    if banned_candidates is None:
        banned_candidates = set()

    t0 = perf_counter()
    model = cp_model.CpModel()
    nodes = reduced["nodes"]
    groups = reduced["unique_groups"]
    prev_set = set(prev_selected)

    cids = sorted(nodes.keys())
    var = {cid: model.NewBoolVar(f"x_{cid}") for cid in cids}

    # Candidate bans from events.
    for cid in banned_candidates:
        if cid in var:
            model.Add(var[cid] == 0)

    # One-hot per subtask.
    selected_prev_by_subtask = {}
    for cid in prev_set:
        sid = nodes[cid]["subtask_id"]
        selected_prev_by_subtask[sid] = cid

    for sid, g in groups.items():
        model.Add(sum(var[cid] for cid in g) == 1)
        # Freeze executed prefix.
        if sid in frozen_subtasks and sid in selected_prev_by_subtask:
            keep = selected_prev_by_subtask[sid]
            if keep in var:
                model.Add(var[keep] == 1)
        # Event-localized repair window: outside editable subtasks, preserve
        # previous assignment when available.
        elif editable_subtasks is not None and sid not in editable_subtasks:
            keep = selected_prev_by_subtask.get(sid)
            if keep in var and keep not in banned_candidates:
                model.Add(var[keep] == 1)

    # Invalid edge constraints.
    for key in ("conflict_edges", "precedence_invalid_edges", "sync_invalid_edges"):
        for u, v in reduced.get(key, []):
            model.Add(var[u] + var[v] <= 1)

    # Objective: base cost + edit penalty wrt previous selection.
    scale = 1000
    obj_terms = []
    constant = 0
    for cid in cids:
        base = int(round(float(nodes[cid]["cost"]) * scale))
        obj_terms.append(base * var[cid])
        if edit_penalty > 0:
            pen = int(round(float(edit_penalty) * scale))
            if cid in prev_set:
                constant += pen
                obj_terms.append(-pen * var[cid])  # penalty*(1-x)
            else:
                obj_terms.append(pen * var[cid])  # penalty*x
    model.Minimize(sum(obj_terms) + constant)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_sec)
    solver.parameters.num_search_workers = int(num_workers)
    status = solver.Solve(model)

    selected = []
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        selected = [cid for cid in cids if solver.Value(var[cid]) == 1]

    runtime = perf_counter() - t0
    feasible = is_feasible(reduced, selected)
    return {
        "selected_candidate_ids": selected,
        "feasible": feasible,
        "objective": objective_from_selection(reduced, selected) if selected else float("inf"),
        "runtime_sec": runtime,
        "status": solver.StatusName(status),
        "editable_subtasks_count": len(editable_subtasks) if editable_subtasks is not None else -1,
    }
