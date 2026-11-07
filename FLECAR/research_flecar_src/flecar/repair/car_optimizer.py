from __future__ import annotations

from time import perf_counter
from typing import Any

from qcma.repair.event_simulator import edit_distance_by_subtask, frozen_subtasks_at_time
from qcma.repair.repair_cpsat import solve_repair_cpsat
from qcma.solvers.common import objective_from_selection

from flecar.dispatch import check_dispatchability
from flecar.envelope import synthesize_envelope
from flecar.repair.normalization import bounded_ratio, clip01


def _expand_localized_set(
    reduced: dict[str, Any],
    localized_subtasks: set[str],
) -> set[str]:
    if not localized_subtasks:
        return set()
    nodes = reduced.get("nodes", {})
    expanded = set(localized_subtasks)

    for pair in reduced.get("precedence_pairs", []):
        if not isinstance(pair, list) or len(pair) != 2:
            continue
        u, v = str(pair[0]), str(pair[1])
        if u in localized_subtasks or v in localized_subtasks:
            expanded.add(u)
            expanded.add(v)

    for edge in reduced.get("conflict_edges", []):
        if not isinstance(edge, (list, tuple)) or len(edge) != 2:
            continue
        su = str(nodes.get(str(edge[0]), {}).get("subtask_id", ""))
        sv = str(nodes.get(str(edge[1]), {}).get("subtask_id", ""))
        if not su or not sv:
            continue
        if su in localized_subtasks or sv in localized_subtasks:
            expanded.add(su)
            expanded.add(sv)

    for edge in reduced.get("sync_invalid_edges", []):
        if not isinstance(edge, (list, tuple)) or len(edge) != 2:
            continue
        su = str(nodes.get(str(edge[0]), {}).get("subtask_id", ""))
        sv = str(nodes.get(str(edge[1]), {}).get("subtask_id", ""))
        if not su or not sv:
            continue
        if su in localized_subtasks or sv in localized_subtasks:
            expanded.add(su)
            expanded.add(sv)
    return expanded


def run_car_local_resynthesis(
    reduced: dict[str, Any],
    prev_selected: list[str],
    *,
    trigger_time: int,
    banned_candidates: set[str],
    localized_subtasks: set[str],
    lambda_e: float,
    lambda_r: float,
    lambda_c: float,
    lambda_m: float,
    base_edit_penalty: float,
    diagnostics: dict[str, Any],
    time_budget_sec: float,
    prev_envelope: dict[str, Any] | None = None,
) -> dict[str, Any]:
    t0_all = perf_counter()

    frozen = frozen_subtasks_at_time(reduced, prev_selected, trigger_time=trigger_time)
    all_subtasks = set(str(x) for x in reduced.get("unique_groups", {}).keys())
    local_editable = set(str(s) for s in localized_subtasks) - set(frozen)
    expanded_editable = _expand_localized_set(reduced, local_editable) - set(frozen)
    global_editable = set(all_subtasks) - set(frozen)

    # Ensure non-empty editable set to avoid infeasible frozen deadlock after bans.
    if not local_editable:
        local_editable = set(global_editable)
    if not expanded_editable:
        expanded_editable = set(global_editable)

    prev_obj = objective_from_selection(reduced, prev_selected) if prev_selected else float("inf")
    prev_ctrl = 0.0
    if prev_envelope is not None:
        prev_ctrl = float(prev_envelope.get("metrics", {}).get("controllability_margin", 0.0))
    else:
        prev_ctrl = float(synthesize_envelope(reduced, prev_selected).get("metrics", {}).get("controllability_margin", 0.0))

    severity = float(diagnostics.get("envelope_violation_severity", 0.0))
    future_conflict_forecast = float(diagnostics.get("future_conflict_forecast", 0.0))
    sync_risk = float(diagnostics.get("synchronized_task_risk", 0.0))
    resource_shift = float(diagnostics.get("resource_criticality_shift", 0.0))

    base_penalty = max(0.1, float(base_edit_penalty) + 0.4 * severity + 0.2 * future_conflict_forecast)

    plan_space = [
        ("localized", local_editable, 1.10),
        ("expanded", expanded_editable, 0.95),
        ("global", global_editable, 0.85),
    ]

    candidates: list[dict[str, Any]] = []
    per_call_budget = max(0.05, float(time_budget_sec) / float(max(1, len(plan_space))))

    for plan_name, editable, pen_factor in plan_space:
        t0 = perf_counter()
        out = solve_repair_cpsat(
            reduced,
            prev_selected,
            frozen_subtasks=frozen,
            banned_candidates=set(str(x) for x in banned_candidates),
            editable_subtasks=set(editable) if editable else None,
            edit_penalty=float(base_penalty) * float(pen_factor),
            time_limit_sec=per_call_budget,
        )
        latency = perf_counter() - t0

        selected = list(out.get("selected_candidate_ids", []))
        if not selected:
            continue

        edits = int(edit_distance_by_subtask(reduced, prev_selected, selected))
        n_subtasks = max(1, len(reduced.get("unique_groups", {})))
        c_edit = clip01(float(edits) / float(n_subtasks))

        new_obj = objective_from_selection(reduced, selected)
        c_replan_gap = float(max(0.0, float(new_obj) - float(prev_obj)) / max(1.0, abs(float(prev_obj))))
        c_replan = float(
            0.7 * bounded_ratio(c_replan_gap, scale=1.0)
            + 0.3 * (0.0 if bool(out.get("feasible", False)) else 1.0)
        )

        env_new = synthesize_envelope(reduced, selected)
        disp = check_dispatchability(env_new)
        new_ctrl = float(disp.get("controllability_margin", 0.0))
        delta_ctrl_raw = float(new_ctrl - prev_ctrl)
        delta_ctrl = float(bounded_ratio(max(0.0, delta_ctrl_raw), scale=1.0))

        future_risk = float(
            0.5 * float(future_conflict_forecast)
            + 0.25 * bounded_ratio(max(0.0, -new_ctrl), scale=1.0)
            + 0.125 * float(sync_risk)
            + 0.125 * float(resource_shift)
        )

        objective = (
            float(lambda_e) * c_edit
            + float(lambda_r) * c_replan
            - float(lambda_c) * delta_ctrl
            + float(lambda_m) * future_risk
        )

        candidates.append(
            {
                "plan_name": plan_name,
                "selected_candidate_ids": selected,
                "feasible": bool(out.get("feasible", False)),
                "status": str(out.get("status", "")),
                "runtime_sec": float(latency),
                "envelope": env_new,
                "dispatchable": bool(disp.get("dispatchable", False)),
                "terms": {
                    "C_edit": float(c_edit),
                    "C_replan": float(c_replan),
                    "DeltaCtrl": float(delta_ctrl),
                    "FutureRisk": float(future_risk),
                    "C_replan_raw": float(c_replan_gap),
                    "DeltaCtrl_raw": float(delta_ctrl_raw),
                },
                "objective": float(objective),
                "edit_burden": int(edits),
                "envelope_churn": float(c_edit),
                "penalty_used": float(base_penalty) * float(pen_factor),
                "editable_subtasks": sorted(str(x) for x in editable),
            }
        )

    if not candidates:
        return {
            "selected_candidate_ids": list(prev_selected),
            "feasible": False,
            "runtime_sec": float(perf_counter() - t0_all),
            "status": "NO_CAR_CANDIDATE",
            "car_meta": {
                "localized_subtasks": sorted(str(x) for x in local_editable),
                "frozen_subtasks": sorted(str(x) for x in frozen),
                "edit_penalty": float(base_penalty),
                "ctrl_term": float(max(0.0, -prev_ctrl)),
                "future_risk": float(future_conflict_forecast),
                "edit_burden": 0,
                "envelope_churn": 0.0,
                "repair_regret": 1.0,
                "best_plan": "none",
                "objective_terms": {
                    "C_edit": 0.0,
                    "C_replan": 1.0,
                    "DeltaCtrl": 0.0,
                    "FutureRisk": float(future_conflict_forecast),
                },
            },
        }

    feasible_dispatchable = [c for c in candidates if c["feasible"] and c["dispatchable"]]
    pool = feasible_dispatchable if feasible_dispatchable else candidates
    pool_sorted = sorted(
        pool,
        key=lambda c: (
            float(c["objective"]),
            0 if c["dispatchable"] else 1,
            int(c["edit_burden"]),
        ),
    )
    best = pool_sorted[0]
    baseline_obj = min(float(c["objective"]) for c in pool_sorted)
    repair_regret = float(best["objective"] - baseline_obj)

    return {
        "selected_candidate_ids": list(best["selected_candidate_ids"]),
        "feasible": bool(best["feasible"]),
        "runtime_sec": float(perf_counter() - t0_all),
        "status": str(best["status"]),
        "envelope": best["envelope"],
        "car_meta": {
            "localized_subtasks": sorted(str(x) for x in local_editable),
            "expanded_subtasks": sorted(str(x) for x in expanded_editable),
            "frozen_subtasks": sorted(str(x) for x in frozen),
            "edit_penalty": float(best["penalty_used"]),
            "ctrl_term": float(max(0.0, -prev_ctrl)),
            "future_risk": float(future_conflict_forecast),
            "edit_burden": int(best["edit_burden"]),
            "envelope_churn": float(best["envelope_churn"]),
            "repair_regret": float(repair_regret),
            "best_plan": str(best["plan_name"]),
            "objective_terms": {
                "C_edit": float(best["terms"]["C_edit"]),
                "C_replan": float(best["terms"]["C_replan"]),
                "DeltaCtrl": float(best["terms"]["DeltaCtrl"]),
                "FutureRisk": float(best["terms"]["FutureRisk"]),
                "C_replan_raw": float(best["terms"]["C_replan_raw"]),
                "DeltaCtrl_raw": float(best["terms"]["DeltaCtrl_raw"]),
            },
            "candidate_objectives": [
                {
                    "plan": str(c["plan_name"]),
                    "objective": float(c["objective"]),
                    "dispatchable": bool(c["dispatchable"]),
                    "feasible": bool(c["feasible"]),
                }
                for c in sorted(candidates, key=lambda x: float(x["objective"]))
            ],
        },
    }
