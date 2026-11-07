"""OR-Tools CP-SAT baseline solver."""

from __future__ import annotations

from time import perf_counter
from typing import Any

from ortools.sat.python import cp_model

from qcma.solvers.common import is_feasible, objective_from_selection


def solve_cpsat(
    reduced: dict[str, Any],
    *,
    time_limit_sec: float = 1.0,
    num_workers: int = 8,
    feasible_first: bool = False,
    feasible_probe_frac: float = 0.35,
    optimize_objective: bool = True,
) -> dict[str, Any]:
    t0 = perf_counter()
    nodes = reduced["nodes"]
    cids = sorted(nodes.keys())

    def _build_model(with_objective: bool) -> tuple[cp_model.CpModel, dict[str, Any]]:
        model = cp_model.CpModel()
        var = {cid: model.NewBoolVar(f"x_{cid}") for cid in cids}

        # One-hot per subtask
        for _, group in reduced["unique_groups"].items():
            if not group:
                continue
            model.Add(sum(var[cid] for cid in group) == 1)

        # Invalid pair constraints
        for key in ("conflict_edges", "precedence_invalid_edges", "sync_invalid_edges"):
            for u, v in reduced.get(key, []):
                model.Add(var[u] + var[v] <= 1)

        if with_objective:
            scale = 1000
            model.Minimize(
                sum(
                    int(round(float(nodes[cid]["cost"]) * scale)) * var[cid]
                    for cid in cids
                )
            )
        return model, var

    selected: list[str] = []
    status_name = "UNKNOWN"

    if not optimize_objective:
        model, var = _build_model(with_objective=False)
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = float(time_limit_sec)
        solver.parameters.num_search_workers = int(num_workers)
        status = solver.Solve(model)
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            selected = [cid for cid in cids if solver.Value(var[cid]) == 1]
        status_name = solver.StatusName(status)
    elif not feasible_first:
        model, var = _build_model(with_objective=True)
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = float(time_limit_sec)
        solver.parameters.num_search_workers = int(num_workers)
        status = solver.Solve(model)
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            selected = [cid for cid in cids if solver.Value(var[cid]) == 1]
        status_name = solver.StatusName(status)
    else:
        # Phase 1: quickly search for any feasible schedule.
        probe_frac = min(max(float(feasible_probe_frac), 0.05), 0.95)
        probe_limit = max(0.01, float(time_limit_sec) * probe_frac)

        model_probe, var_probe = _build_model(with_objective=False)
        solver_probe = cp_model.CpSolver()
        solver_probe.parameters.max_time_in_seconds = probe_limit
        solver_probe.parameters.num_search_workers = int(num_workers)
        status_probe = solver_probe.Solve(model_probe)

        probe_selected: list[str] = []
        if status_probe in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            probe_selected = [cid for cid in cids if solver_probe.Value(var_probe[cid]) == 1]

        elapsed = perf_counter() - t0
        remain = max(0.01, float(time_limit_sec) - elapsed)

        # Phase 2: optimize objective with remaining budget, warm-started from probe.
        model_obj, var_obj = _build_model(with_objective=True)
        if probe_selected:
            probe_set = set(probe_selected)
            for cid in cids:
                model_obj.AddHint(var_obj[cid], 1 if cid in probe_set else 0)

        solver_obj = cp_model.CpSolver()
        solver_obj.parameters.max_time_in_seconds = remain
        solver_obj.parameters.num_search_workers = int(num_workers)
        status_obj = solver_obj.Solve(model_obj)
        if status_obj in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            selected = [cid for cid in cids if solver_obj.Value(var_obj[cid]) == 1]
            status_name = solver_obj.StatusName(status_obj)
        elif probe_selected:
            selected = list(probe_selected)
            status_name = f"{solver_probe.StatusName(status_probe)}_PROBE"
        else:
            status_name = solver_obj.StatusName(status_obj)

    runtime = perf_counter() - t0
    feasible = is_feasible(reduced, selected)
    objective = objective_from_selection(reduced, selected) if selected else float("inf")
    return {
        "solver": "cpsat",
        "selected_candidate_ids": selected,
        "feasible": feasible,
        "objective": objective,
        "runtime_sec": runtime,
        "solver_status": status_name,
    }
