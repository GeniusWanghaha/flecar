"""Static benchmark metrics for QCMA solvers."""

from __future__ import annotations

from typing import Any

from qcma.solvers.common import complete_selection, violation_counts
from qcma.solvers.common import objective_from_selection


def evaluate_static_solution(
    reduced: dict[str, Any], solver_output: dict[str, Any]
) -> dict[str, Any]:
    selected = solver_output.get("selected_candidate_ids", [])
    vio = violation_counts(reduced, selected)
    feasible = sum(vio.values()) == 0
    complete = complete_selection(reduced, selected)

    objective_cost = (
        objective_from_selection(reduced, selected) if selected else float("inf")
    )
    return {
        "solver": solver_output["solver"],
        "episode_id": reduced["episode_id"],
        "task_text": reduced["task_text"],
        "runtime_sec": float(solver_output.get("runtime_sec", 0.0)),
        "objective": float(objective_cost),
        "feasible": bool(feasible),
        "complete": bool(complete),
        "n_selected": len(selected),
        "n_subtasks": len(reduced["unique_groups"]),
        "violations_unique_assignment": int(vio["unique_assignment"]),
        "violations_conflict": int(vio["conflict"]),
        "violations_precedence": int(vio["precedence"]),
        "violations_sync": int(vio["sync"]),
        "violations_total": int(sum(vio.values())),
        "solver_meta": {
            k: v
            for k, v in solver_output.items()
            if k not in {"solver", "selected_candidate_ids", "runtime_sec", "objective", "feasible"}
        },
    }


def aggregate_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_solver: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_solver.setdefault(row["solver"], []).append(row)

    out: list[dict[str, Any]] = []
    for solver, vals in sorted(by_solver.items()):
        n = len(vals)
        feasible_rate = sum(
            1 for v in vals if str(v["feasible"]).strip().lower() in {"1", "true", "yes"}
        ) / max(1, n)
        feasible_objs = [
            float(v["objective"])
            for v in vals
            if str(v["feasible"]).strip().lower() in {"1", "true", "yes"}
        ]
        mean_obj = (
            sum(feasible_objs) / max(1, len(feasible_objs))
            if feasible_objs
            else float("inf")
        )
        mean_runtime = sum(float(v["runtime_sec"]) for v in vals) / max(1, n)
        mean_vio = sum(float(v["violations_total"]) for v in vals) / max(1, n)
        out.append(
            {
                "solver": solver,
                "n_instances": n,
                "feasible_rate": feasible_rate,
                "mean_objective_feasible_only": mean_obj,
                "mean_runtime_sec": mean_runtime,
                "mean_violations_total": mean_vio,
            }
        )
    return out
