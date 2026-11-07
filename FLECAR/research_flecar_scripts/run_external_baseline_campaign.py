from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from time import perf_counter
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC_Q = ROOT / "src"
SRC_F = ROOT / "research_flecar" / "src"
for p in [SRC_Q, SRC_F]:
    if str(p) not in sys.path:
        sys.path.append(str(p))

from qcma.eval.metrics_static import evaluate_static_solution  # noqa: E402
from qcma.graph.reduced_graph import build_reduced_graph  # noqa: E402
from qcma.repair.event_simulator import (  # noqa: E402
    EVENT_TYPES,
    adaptive_scope_controller,
    apply_event_to_bans,
    edit_distance_by_subtask,
    frozen_subtasks_at_time,
    sample_trigger_times,
)
from qcma.repair.repair_cpsat import solve_repair_cpsat  # noqa: E402
from qcma.solvers.common import is_feasible  # noqa: E402
from qcma.solvers.cpsat_runner import solve_cpsat  # noqa: E402

from flecar.dispatch import dispatch_step  # noqa: E402
from flecar.envelope import synthesize_envelope  # noqa: E402
from flecar.evaluation.common_metrics import common_next_event_survivability, deterministic_rng, normalized_repair_burden  # noqa: E402
from flecar.repair import event_diagnostics  # noqa: E402
from flecar.utils import list_instance_files, read_json, write_csv, write_json  # noqa: E402


@dataclass
class CampaignConfig:
    top_k: int = 2
    density_scale: float = 1.5
    time_budget_sec: float = 0.5
    seeds: tuple[int, ...] = (7, 11, 19, 23, 29)
    max_static_instances: int = 300
    max_online_instances: int = 50
    max_chained_instances: int = 40
    events_per_instance: int = 4


BASELINES = [
    "external_temporal_dispatch",
    "external_temporal_replan",
    "external_dispatch_suffix",
    "external_stnu_guarded_dispatch",
    "external_rolling_horizon_engine",
]


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _to_float(x: Any) -> float:
    if x is None:
        return 0.0
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none"}:
        return 0.0
    try:
        return float(s)
    except Exception:
        return 0.0


def _bootstrap_ci(vals: list[float], n_boot: int = 2000, alpha: float = 0.05, seed: int = 7) -> tuple[float, float]:
    if not vals:
        return 0.0, 0.0
    rng = random.Random(seed)
    n = len(vals)
    boots: list[float] = []
    for _ in range(n_boot):
        sample = [vals[rng.randrange(n)] for _ in range(n)]
        boots.append(sum(sample) / max(1, len(sample)))
    boots.sort()
    lo = boots[max(0, int((alpha / 2) * len(boots)) - 1)]
    hi = boots[min(len(boots) - 1, int((1 - alpha / 2) * len(boots)) - 1)]
    return float(lo), float(hi)


def _exact_sign_p(wins: int, losses: int) -> float:
    n = wins + losses
    if n <= 0:
        return 1.0
    k = min(wins, losses)
    p = 0.0
    for i in range(k + 1):
        p += math.comb(n, i)
    return min(1.0, 2.0 * p / (2**n))


def _paired_stats(a_vals: list[float], b_vals: list[float], seed: int = 7) -> dict[str, Any]:
    n = min(len(a_vals), len(b_vals))
    if n == 0:
        return {
            "n_pairs": 0,
            "mean_a": 0.0,
            "mean_b": 0.0,
            "paired_delta": 0.0,
            "delta_ci95_low": 0.0,
            "delta_ci95_high": 0.0,
            "wins": 0,
            "losses": 0,
            "ties": 0,
            "p_two_sided_exact_sign_test": 1.0,
            "std_a": 0.0,
            "std_b": 0.0,
            "stderr_a": 0.0,
            "stderr_b": 0.0,
        }
    a = a_vals[:n]
    b = b_vals[:n]
    deltas = [x - y for x, y in zip(a, b)]
    wins = sum(1 for d in deltas if d > 1e-12)
    losses = sum(1 for d in deltas if d < -1e-12)
    ties = n - wins - losses
    lo, hi = _bootstrap_ci(deltas, seed=seed)
    std_a = pstdev(a) if len(a) > 1 else 0.0
    std_b = pstdev(b) if len(b) > 1 else 0.0
    return {
        "n_pairs": n,
        "mean_a": float(mean(a)),
        "mean_b": float(mean(b)),
        "paired_delta": float(mean(deltas)),
        "delta_ci95_low": float(lo),
        "delta_ci95_high": float(hi),
        "wins": int(wins),
        "losses": int(losses),
        "ties": int(ties),
        "p_two_sided_exact_sign_test": float(_exact_sign_p(wins, losses)),
        "std_a": float(std_a),
        "std_b": float(std_b),
        "stderr_a": float(std_a / math.sqrt(max(1, len(a)))),
        "stderr_b": float(std_b / math.sqrt(max(1, len(b)))),
    }


def _normalize_event_type(event_type: str) -> str:
    if event_type == "temporary_agent_unavailability":
        return "agent_unavailable"
    return str(event_type)


def _init_campaign_states(reduced: dict[str, Any], selected: list[str]) -> dict[str, dict[str, Any]]:
    return {
        "external_temporal_dispatch": {
            "selected": list(selected),
            "envelope": synthesize_envelope(reduced, list(selected), slack_expand_ratio=0.12, min_window_pad=1, optimize_grid=False),
        },
        "external_temporal_replan": {
            "selected": list(selected),
            "envelope": None,
        },
        "external_dispatch_suffix": {
            "selected": list(selected),
            "envelope": synthesize_envelope(reduced, list(selected), slack_expand_ratio=0.25, min_window_pad=2, optimize_grid=False),
        },
        "external_stnu_guarded_dispatch": {
            "selected": list(selected),
            "envelope": synthesize_envelope(reduced, list(selected), slack_expand_ratio=0.18, min_window_pad=2, optimize_grid=True),
        },
        "external_rolling_horizon_engine": {
            "selected": list(selected),
            "envelope": synthesize_envelope(reduced, list(selected), slack_expand_ratio=0.10, min_window_pad=1, optimize_grid=False),
        },
    }


def _run_event_step(
    method: str,
    reduced: dict[str, Any],
    st: dict[str, Any],
    *,
    trigger_time: int,
    event_type: str,
    banned_set: set[str],
    cfg: CampaignConfig,
    stnu_threshold: float = 0.20,
) -> dict[str, Any]:
    prev_selected = list(st["selected"])
    env = st.get("envelope")

    repair_latency = 0.0
    edits = 0
    repair_regret = 0.0
    envelope_churn = 0.0
    repair_free_absorption = 0
    ctrl_preserve = 0
    repair_triggered = 0

    if method == "external_temporal_replan":
        frozen = frozen_subtasks_at_time(reduced, prev_selected, trigger_time=trigger_time)
        t0 = perf_counter()
        out = solve_repair_cpsat(
            reduced,
            prev_selected,
            frozen_subtasks=frozen,
            banned_candidates=banned_set,
            editable_subtasks=None,
            edit_penalty=0.0,
            time_limit_sec=float(cfg.time_budget_sec),
        )
        repair_latency = perf_counter() - t0
        new_selected = list(out.get("selected_candidate_ids", []))
        edits = int(edit_distance_by_subtask(reduced, prev_selected, new_selected))
        envelope_churn = float(edits) / float(max(1, len(reduced.get("unique_groups", {}))))
        repair_regret = envelope_churn + (0.0 if out.get("feasible", False) else 1.0)
        repair_triggered = 1

    elif method == "external_temporal_dispatch":
        disp = dispatch_step(
            reduced,
            env,
            prev_selected,
            banned_candidates=banned_set,
            current_time=int(trigger_time),
        )
        new_selected = list(disp["selection"])
        repair_free_absorption = int(disp["dispatchable"])
        ctrl_preserve = int(float(disp.get("controllability_margin", 0.0)) >= 0.0)
        env = disp.get("effective_envelope", env)

        if not bool(disp["dispatchable"]):
            frozen = frozen_subtasks_at_time(reduced, prev_selected, trigger_time=trigger_time)
            t0 = perf_counter()
            out = solve_repair_cpsat(
                reduced,
                prev_selected,
                frozen_subtasks=frozen,
                banned_candidates=banned_set,
                editable_subtasks=None,
                edit_penalty=4.0,
                time_limit_sec=float(cfg.time_budget_sec),
            )
            repair_latency = perf_counter() - t0
            new_selected = list(out.get("selected_candidate_ids", []))
            edits = int(edit_distance_by_subtask(reduced, prev_selected, new_selected))
            envelope_churn = float(edits) / float(max(1, len(reduced.get("unique_groups", {}))))
            repair_regret = envelope_churn + (0.0 if out.get("feasible", False) else 1.0)
            repair_triggered = 1
            env = synthesize_envelope(
                reduced,
                new_selected,
                slack_expand_ratio=0.12,
                min_window_pad=1,
                optimize_grid=False,
            )
            ctrl_preserve = int(float(env.get("metrics", {}).get("controllability_margin", 0.0)) >= 0.0)

    elif method == "external_dispatch_suffix":
        disp = dispatch_step(
            reduced,
            env,
            prev_selected,
            banned_candidates=banned_set,
            current_time=int(trigger_time),
        )
        new_selected = list(disp["selection"])
        repair_free_absorption = int(disp["dispatchable"])
        ctrl_preserve = int(float(disp.get("controllability_margin", 0.0)) >= 0.0)
        env = disp.get("effective_envelope", env)

        if not bool(disp["dispatchable"]):
            ctrl = adaptive_scope_controller(
                reduced,
                prev_selected,
                trigger_time=int(trigger_time),
                banned_candidates=banned_set,
                event_type=_normalize_event_type(event_type),
                base_edit_penalty=6.0,
                default_radius=1,
            )
            t0 = perf_counter()
            out = solve_repair_cpsat(
                reduced,
                prev_selected,
                frozen_subtasks=set(ctrl["frozen_subtasks"]),
                banned_candidates=banned_set,
                editable_subtasks=set(ctrl["editable_subtasks"]),
                edit_penalty=float(ctrl["edit_penalty"]),
                time_limit_sec=float(cfg.time_budget_sec),
            )
            repair_latency = perf_counter() - t0
            new_selected = list(out.get("selected_candidate_ids", []))
            edits = int(edit_distance_by_subtask(reduced, prev_selected, new_selected))
            envelope_churn = float(edits) / float(max(1, len(reduced.get("unique_groups", {}))))
            repair_regret = envelope_churn + (0.0 if out.get("feasible", False) else 1.0)
            repair_triggered = 1
            env = synthesize_envelope(
                reduced,
                new_selected,
                slack_expand_ratio=0.25,
                min_window_pad=2,
                optimize_grid=False,
            )
            ctrl_preserve = int(float(env.get("metrics", {}).get("controllability_margin", 0.0)) >= 0.0)

    elif method == "external_stnu_guarded_dispatch":
        disp = dispatch_step(
            reduced,
            env,
            prev_selected,
            banned_candidates=banned_set,
            current_time=int(trigger_time),
        )
        new_selected = list(disp["selection"])
        env = disp.get("effective_envelope", env)
        ctrl_margin = float(disp.get("controllability_margin", 0.0))

        diag = event_diagnostics(
            reduced,
            env,
            prev_selected,
            event_type=event_type,
            trigger_time=int(trigger_time),
            banned_candidates=banned_set,
        )
        # Published-inspired stronger dispatch gate:
        # require both dispatchability and controllability margin.
        allow_dispatch = bool(disp["dispatchable"]) and ctrl_margin >= float(stnu_threshold) and bool(diag["dispatchable_after_event"])
        if allow_dispatch:
            repair_free_absorption = 1
            ctrl_preserve = int(ctrl_margin >= 0.0)
        else:
            ctrl = adaptive_scope_controller(
                reduced,
                prev_selected,
                trigger_time=int(trigger_time),
                banned_candidates=banned_set,
                event_type=_normalize_event_type(event_type),
                base_edit_penalty=5.0,
                default_radius=2,
            )
            t0 = perf_counter()
            out = solve_repair_cpsat(
                reduced,
                prev_selected,
                frozen_subtasks=set(ctrl["frozen_subtasks"]),
                banned_candidates=banned_set,
                editable_subtasks=set(ctrl["editable_subtasks"]),
                edit_penalty=max(1.5, float(ctrl["edit_penalty"]) - 1.0),
                time_limit_sec=float(cfg.time_budget_sec),
            )
            repair_latency = perf_counter() - t0
            new_selected = list(out.get("selected_candidate_ids", []))
            edits = int(edit_distance_by_subtask(reduced, prev_selected, new_selected))
            envelope_churn = float(edits) / float(max(1, len(reduced.get("unique_groups", {}))))
            repair_regret = envelope_churn + (0.0 if out.get("feasible", False) else 1.0)
            repair_triggered = 1
            env = synthesize_envelope(
                reduced,
                new_selected,
                slack_expand_ratio=0.18,
                min_window_pad=2,
                optimize_grid=True,
            )
            ctrl_preserve = int(float(env.get("metrics", {}).get("controllability_margin", 0.0)) >= 0.0)

    elif method == "external_rolling_horizon_engine":
        # Rolling-horizon temporal engine: always replan locally every event.
        ctrl = adaptive_scope_controller(
            reduced,
            prev_selected,
            trigger_time=int(trigger_time),
            banned_candidates=banned_set,
            event_type=_normalize_event_type(event_type),
            base_edit_penalty=2.5,
            default_radius=2,
        )
        t0 = perf_counter()
        out = solve_repair_cpsat(
            reduced,
            prev_selected,
            frozen_subtasks=set(ctrl["frozen_subtasks"]),
            banned_candidates=banned_set,
            editable_subtasks=set(ctrl["editable_subtasks"]),
            edit_penalty=max(0.5, float(ctrl["edit_penalty"]) - 2.0),
            time_limit_sec=float(cfg.time_budget_sec),
        )
        repair_latency = perf_counter() - t0
        new_selected = list(out.get("selected_candidate_ids", []))
        edits = int(edit_distance_by_subtask(reduced, prev_selected, new_selected))
        envelope_churn = float(edits) / float(max(1, len(reduced.get("unique_groups", {}))))
        repair_regret = envelope_churn + (0.0 if out.get("feasible", False) else 1.0)
        repair_triggered = 1
        env = synthesize_envelope(
            reduced,
            new_selected,
            slack_expand_ratio=0.10,
            min_window_pad=1,
            optimize_grid=False,
        )
        ctrl_preserve = int(float(env.get("metrics", {}).get("controllability_margin", 0.0)) >= 0.0)

    else:
        raise ValueError(f"unknown method {method}")

    st["selected"] = list(new_selected)
    st["envelope"] = env

    return {
        "selected": list(new_selected),
        "repair_latency": float(repair_latency),
        "edit_burden": int(edits),
        "repair_regret": float(repair_regret),
        "envelope_churn": float(envelope_churn),
        "repair_free_absorption": int(repair_free_absorption),
        "controllability_preserved": int(ctrl_preserve),
        "repair_triggered": int(repair_triggered),
        "controllability_margin": float(env.get("metrics", {}).get("controllability_margin", 0.0)) if env else 0.0,
    }


def run_static_campaign(instance_files: list[Path], cfg: CampaignConfig) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in cfg.seeds:
        for p in instance_files:
            inst = read_json(p)
            eid = str(inst["episode_id"])
            reduced = build_reduced_graph(
                inst,
                top_k=int(cfg.top_k),
                constraint_density_scale=float(cfg.density_scale),
                density_seed=int(seed),
                candidate_rank_mode="cost_start_duration",
                enable_dominated_pruning=True,
            )
            out = solve_cpsat(reduced, time_limit_sec=float(cfg.time_budget_sec))
            selected = list(out.get("selected_candidate_ids", []))
            met = evaluate_static_solution(reduced, out)

            for method in BASELINES:
                env = None
                if method == "external_temporal_dispatch":
                    env = synthesize_envelope(reduced, selected, slack_expand_ratio=0.12, min_window_pad=1, optimize_grid=False)
                elif method == "external_dispatch_suffix":
                    env = synthesize_envelope(reduced, selected, slack_expand_ratio=0.25, min_window_pad=2, optimize_grid=False)
                elif method == "external_stnu_guarded_dispatch":
                    env = synthesize_envelope(reduced, selected, slack_expand_ratio=0.18, min_window_pad=2, optimize_grid=True)
                elif method == "external_rolling_horizon_engine":
                    env = synthesize_envelope(reduced, selected, slack_expand_ratio=0.10, min_window_pad=1, optimize_grid=False)

                em = env.get("metrics", {}) if env else {}
                rows.append(
                    {
                        "seed": int(seed),
                        "episode_id": eid,
                        "method": method,
                        "feasible": int(bool(met["feasible"])),
                        "violations_total": int(met["violations_total"]),
                        "runtime_sec": float(met["runtime_sec"]),
                        "objective": float(met["objective"]),
                        "dispatchability_success": int(bool(em.get("dispatchability_success", False))),
                        "critical_slack_margin": float(em.get("critical_slack_margin", 0.0)),
                        "absorbable_disturbance_radius": float(em.get("controllability_margin", 0.0) if env else 0.0),
                        "flexibility_volume": float(em.get("flexibility_volume", 0.0)),
                        "n_subtasks": int(len(reduced.get("unique_groups", {}))),
                        "n_conflict_edges": int(len(reduced.get("conflict_edges", []))),
                        "candidate_density": float(len(reduced.get("nodes", {})) / max(1, len(reduced.get("unique_groups", {})))),
                    }
                )
    return rows


def run_online_campaign(
    instance_files: list[Path],
    cfg: CampaignConfig,
    *,
    chain_len: int | None = None,
    stnu_threshold: float = 0.20,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in cfg.seeds:
        for p in instance_files:
            inst = read_json(p)
            eid = str(inst["episode_id"])
            reduced = build_reduced_graph(
                inst,
                top_k=int(cfg.top_k),
                constraint_density_scale=float(cfg.density_scale),
                density_seed=int(seed),
                candidate_rank_mode="cost_start_duration",
                enable_dominated_pruning=True,
            )
            out = solve_cpsat(reduced, time_limit_sec=float(cfg.time_budget_sec))
            selected0 = list(out.get("selected_candidate_ids", []))
            states = _init_campaign_states(reduced, selected0)

            n_events = int(chain_len) if chain_len is not None else int(cfg.events_per_instance)
            trigger_times = sample_trigger_times(reduced, selected0, n_events)

            for ev_idx, t in enumerate(trigger_times):
                event_type = str(EVENT_TYPES[ev_idx % len(EVENT_TYPES)])
                for method in BASELINES:
                    st = states[method]
                    prev_selected = list(st["selected"])
                    event_rng = deterministic_rng("external_online_event", seed, eid, ev_idx, int(t), event_type)
                    bans = apply_event_to_bans(
                        reduced,
                        prev_selected,
                        trigger_time=int(t),
                        event_type=_normalize_event_type(event_type),
                        rng=event_rng,
                    )
                    banned_set = {str(x) for x in bans}

                    rr = _run_event_step(
                        method,
                        reduced,
                        st,
                        trigger_time=int(t),
                        event_type=event_type,
                        banned_set=banned_set,
                        cfg=cfg,
                        stnu_threshold=float(stnu_threshold),
                    )

                    feasible = int(is_feasible(reduced, rr["selected"]))
                    n_subtasks = len(reduced.get("unique_groups", {}))
                    frozen_now = frozen_subtasks_at_time(reduced, rr["selected"], trigger_time=int(t))
                    cumulative_completion = min(1.0, (len(frozen_now) + feasible) / float(max(1, n_subtasks)))
                    burden = normalized_repair_burden(
                        reduced,
                        prev_selected,
                        rr["selected"],
                        feasible=bool(feasible),
                        repair_latency=float(rr["repair_latency"]),
                        repair_triggered=int(rr["repair_triggered"]),
                        time_budget_sec=float(cfg.time_budget_sec),
                    )
                    future_rng = deterministic_rng("external_future_probe", seed, eid, ev_idx, int(t), event_type)
                    future_feas = common_next_event_survivability(
                        reduced,
                        rr["selected"],
                        trigger_time=int(t),
                        rng=future_rng,
                    )

                    row = {
                        "seed": int(seed),
                        "episode_id": eid,
                        "event_idx": int(ev_idx),
                        "event_type": str(event_type),
                        "trigger_time": int(t),
                        "method": method,
                        "post_event_feasible": int(feasible),
                        "cumulative_completion": float(cumulative_completion),
                        "terminal_success_proxy": int(feasible if ev_idx == len(trigger_times) - 1 else 0),
                        "terminal_completion_proxy": float(cumulative_completion if ev_idx == len(trigger_times) - 1 else 0.0),
                        "repair_free_absorption": int(rr["repair_free_absorption"]),
                        "controllability_preserved": int(rr["controllability_preserved"]),
                        "future_feasibility_after_next_event": int(future_feas),
                        "one_step_survivability": int(future_feas),
                        "repair_regret": float(burden["repair_burden"]),
                        "repair_burden": float(burden["repair_burden"]),
                        "envelope_churn": float(rr["envelope_churn"]),
                        "edit_burden": int(rr["edit_burden"]),
                        "repair_latency": float(rr["repair_latency"]),
                        "controllability_margin": float(rr["controllability_margin"]),
                        "repair_triggered": int(rr["repair_triggered"]),
                    }
                    if chain_len is not None:
                        row["chain_len"] = int(chain_len)
                    row["stnu_threshold"] = float(stnu_threshold)
                    rows.append(row)
    return rows


def _aggregate_by_method(rows: list[dict[str, Any]], fields: list[str]) -> list[dict[str, Any]]:
    by: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by[str(r["method"])].append(r)
    out: list[dict[str, Any]] = []
    for m, vals in sorted(by.items()):
        row = {"method": m, "n_rows": len(vals)}
        for f in fields:
            row[f] = float(sum(_to_float(v.get(f, 0.0)) for v in vals) / max(1, len(vals)))
        out.append(row)
    return out


def _keyed(rows: list[dict[str, Any]], key_fields: tuple[str, ...], value_field: str) -> dict[tuple[str, ...], float]:
    out: dict[tuple[str, ...], float] = {}
    for r in rows:
        k = tuple(str(r.get(kf, "")) for kf in key_fields)
        out[k] = _to_float(r.get(value_field, 0.0))
    return out


def _paired_vs_reference(
    method_rows: list[dict[str, Any]],
    ref_rows: list[dict[str, Any]],
    *,
    key_fields: tuple[str, ...],
    metric_field: str,
    seed: int = 7,
) -> dict[str, Any]:
    m = _keyed(method_rows, key_fields, metric_field)
    r = _keyed(ref_rows, key_fields, metric_field)
    keys = sorted(set(m.keys()).intersection(r.keys()))
    a = [m[k] for k in keys]
    b = [r[k] for k in keys]
    return _paired_stats(a, b, seed=seed)


def export_tables(
    *,
    out_tables: Path,
    static_rows: list[dict[str, Any]],
    online_rows: list[dict[str, Any]],
    chained_rows: list[dict[str, Any]],
    static_ref_rows: list[dict[str, Any]],
    online_ref_rows: list[dict[str, Any]],
    chained_ref_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    out_tables.mkdir(parents=True, exist_ok=True)

    # static main
    static_table: list[dict[str, Any]] = []
    for method in BASELINES:
        mr = [r for r in static_rows if r["method"] == method]
        rr = [r for r in static_ref_rows if r["method"] == "fes"]
        ps = _paired_vs_reference(rr, mr, key_fields=("seed", "episode_id"), metric_field="feasible", seed=7)
        static_table.append(
            {
                "baseline_method": method,
                "comparison": "fes_vs_external_baseline",
                "metric": "feasible",
                **ps,
            }
        )
    write_csv(out_tables / "table_external_baseline_static_main.csv", static_table)

    # online main metrics
    online_table: list[dict[str, Any]] = []
    metrics = [
        "post_event_feasible",
        "future_feasibility_after_next_event",
        "repair_regret",
        "repair_burden",
        "envelope_churn",
        "edit_burden",
    ]
    for method in BASELINES:
        mr_all = [r for r in online_rows if r["method"] == method]
        rr_all = [r for r in online_ref_rows if r["method"] == "fes_car"]
        for metric in metrics:
            ps = _paired_vs_reference(
                rr_all,
                mr_all,
                key_fields=("seed", "episode_id", "event_idx"),
                metric_field=metric,
                seed=7,
            )
            online_table.append(
                {
                    "baseline_method": method,
                    "comparison": "fes_car_vs_external_baseline",
                    "metric": metric,
                    **ps,
                }
            )

        # terminal success uses only terminal rows
        mr_term = [r for r in mr_all if int(r["terminal_success_proxy"]) in (0, 1) and int(r["event_idx"]) == 3]
        rr_term = [r for r in rr_all if int(r["event_idx"]) == 3]
        ps_t = _paired_vs_reference(
            rr_term,
            mr_term,
            key_fields=("seed", "episode_id", "event_idx"),
            metric_field="terminal_success_proxy",
            seed=7,
        )
        online_table.append(
            {
                "baseline_method": method,
                "comparison": "fes_car_vs_external_baseline",
                "metric": "terminal_success_proxy",
                **ps_t,
            }
        )

    write_csv(out_tables / "table_external_baseline_online_main.csv", online_table)

    # harder split: static feasible split + online difficult slices from static metadata mapping
    harder_table: list[dict[str, Any]] = []
    # derive split thresholds from static rows
    n_sub = sorted(_to_float(r["n_subtasks"]) for r in static_rows)
    n_conf = sorted(_to_float(r["n_conflict_edges"]) for r in static_rows)
    dens = sorted(_to_float(r["candidate_density"]) for r in static_rows)
    def q(vals, qq):
        if not vals:
            return 0.0
        idx = int(round((len(vals)-1)*qq))
        return float(vals[idx])
    th_sub = max(2.0, q(n_sub, 0.85))
    th_conf = max(5.0, q(n_conf, 0.85))
    th_den = q(dens, 0.80)

    # map episode to slices per seed from static rows
    epi_slice: dict[tuple[str, str], dict[str, bool]] = {}
    for r in static_rows:
        if r["method"] != BASELINES[0]:
            continue
        key = (str(r["seed"]), str(r["episode_id"]))
        epi_slice[key] = {
            "larger_graph": _to_float(r["n_subtasks"]) >= th_sub,
            "higher_conflict": _to_float(r["n_conflict_edges"]) >= th_conf,
            "higher_candidate_density": _to_float(r["candidate_density"]) >= th_den,
        }

    for method in BASELINES:
        mr = [r for r in static_rows if r["method"] == method]
        rr = [r for r in static_ref_rows if r["method"] == "fes"]
        for sl in ["larger_graph", "higher_conflict", "higher_candidate_density"]:
            mr_s = [r for r in mr if epi_slice.get((str(r["seed"]), str(r["episode_id"])), {}).get(sl, False)]
            rr_s = [r for r in rr if epi_slice.get((str(r["seed"]), str(r["episode_id"])), {}).get(sl, False)]
            ps = _paired_vs_reference(
                rr_s,
                mr_s,
                key_fields=("seed", "episode_id"),
                metric_field="feasible",
                seed=7,
            )
            harder_table.append(
                {
                    "baseline_method": method,
                    "slice": sl,
                    "metric": "feasible",
                    **ps,
                }
            )

    write_csv(out_tables / "table_external_baseline_harder_split.csv", harder_table)

    # chained
    chained_table: list[dict[str, Any]] = []
    for method in BASELINES:
        mr = [r for r in chained_rows if r["method"] == method]
        rr = [r for r in chained_ref_rows if r["method"] == "fes_car"]
        for cl in [2, 3, 4]:
            mr_c = [r for r in mr if int(r["chain_len"]) == cl and int(r["event_idx"]) == cl - 1]
            rr_c = [r for r in rr if int(r["chain_len"]) == cl and int(r["event_idx"]) == cl - 1]
            ps = _paired_vs_reference(
                rr_c,
                mr_c,
                key_fields=("seed", "episode_id", "chain_len", "event_idx"),
                metric_field="terminal_success_proxy",
                seed=7,
            )
            chained_table.append(
                {
                    "baseline_method": method,
                    "chain_len": cl,
                    "metric": "terminal_success_proxy",
                    **ps,
                }
            )
    write_csv(out_tables / "table_external_baseline_chained.csv", chained_table)

    # master summary
    master: list[dict[str, Any]] = []
    for method in BASELINES:
        st = next(r for r in static_table if r["baseline_method"] == method and r["metric"] == "feasible")
        on = next(r for r in online_table if r["baseline_method"] == method and r["metric"] == "post_event_feasible")
        fu = next(r for r in online_table if r["baseline_method"] == method and r["metric"] == "future_feasibility_after_next_event")
        te = next(r for r in online_table if r["baseline_method"] == method and r["metric"] == "terminal_success_proxy")
        ch3 = next(r for r in chained_table if r["baseline_method"] == method and int(r["chain_len"]) == 3)
        master.append(
            {
                "baseline_method": method,
                "static_feasible_delta_fes_minus_baseline": st["paired_delta"],
                "online_post_event_delta_fes_car_minus_baseline": on["paired_delta"],
                "online_future_delta_fes_car_minus_baseline": fu["paired_delta"],
                "online_terminal_delta_fes_car_minus_baseline": te["paired_delta"],
                "chain3_terminal_delta_fes_car_minus_baseline": ch3["paired_delta"],
                "n_pairs_static": st["n_pairs"],
                "n_pairs_online": on["n_pairs"],
                "n_pairs_chain3": ch3["n_pairs"],
            }
        )
    write_csv(out_tables / "table_external_baseline_master_summary.csv", master)

    return {
        "static": static_table,
        "online": online_table,
        "harder": harder_table,
        "chained": chained_table,
        "master": master,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances-dir", type=Path, default=Path("research_flecar/data_build/flecar_instances_v1"))
    ap.add_argument("--out-root", type=Path, default=Path("research_flecar/outputs_reviewer_strengthening"))
    ap.add_argument("--reference-root", type=Path, default=Path("research_flecar/outputs_post_cleanup/raw"))
    ap.add_argument("--seeds", type=str, default="7,11,19,23,29")
    ap.add_argument("--top-k", type=int, default=2)
    ap.add_argument("--density-scale", type=float, default=1.5)
    ap.add_argument("--time-budget-sec", type=float, default=0.5)
    ap.add_argument("--max-static-instances", type=int, default=300)
    ap.add_argument("--max-online-instances", type=int, default=50)
    ap.add_argument("--max-chained-instances", type=int, default=40)
    ap.add_argument("--events-per-instance", type=int, default=4)
    ap.add_argument("--stnu-threshold", type=float, default=0.20)
    args = ap.parse_args()

    seeds = tuple(int(x.strip()) for x in str(args.seeds).split(",") if x.strip())
    cfg = CampaignConfig(
        top_k=int(args.top_k),
        density_scale=float(args.density_scale),
        time_budget_sec=float(args.time_budget_sec),
        seeds=seeds,
        max_static_instances=int(args.max_static_instances),
        max_online_instances=int(args.max_online_instances),
        max_chained_instances=int(args.max_chained_instances),
        events_per_instance=int(args.events_per_instance),
    )

    out_root = ROOT / args.out_root
    raw_dir = out_root / "raw"
    tables_dir = out_root / "exports" / "tables"
    raw_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    files_all = list_instance_files(ROOT / args.instances_dir)
    static_files = files_all[: cfg.max_static_instances]
    online_files = files_all[: cfg.max_online_instances]
    chained_files = files_all[: cfg.max_chained_instances]

    static_rows = run_static_campaign(static_files, cfg)
    online_rows = run_online_campaign(online_files, cfg, stnu_threshold=float(args.stnu_threshold))
    chained_rows = []
    for cl in [2, 3, 4]:
        chained_rows.extend(run_online_campaign(chained_files, cfg, chain_len=cl, stnu_threshold=float(args.stnu_threshold)))

    write_csv(raw_dir / "external_baseline_static_rows.csv", static_rows)
    write_csv(raw_dir / "external_baseline_online_rows.csv", online_rows)
    write_csv(raw_dir / "external_baseline_chained_rows.csv", chained_rows)

    # reference rows from source-of-truth raw outputs
    reference_root = ROOT / args.reference_root
    static_ref_rows = _read_csv(reference_root / "static" / "static_rows.csv")
    online_ref_rows = _read_csv(reference_root / "online" / "online_event_rows.csv")
    chained_ref_rows = _read_csv(reference_root / "chained" / "chained_event_rows.csv")

    tables = export_tables(
        out_tables=tables_dir,
        static_rows=static_rows,
        online_rows=online_rows,
        chained_rows=chained_rows,
        static_ref_rows=static_ref_rows,
        online_ref_rows=online_ref_rows,
        chained_ref_rows=chained_ref_rows,
    )

    manifest = {
        "run_id": "flecar_multi_external_baseline_campaign_v1",
        "config": {
            "top_k": cfg.top_k,
            "density_scale": cfg.density_scale,
            "time_budget_sec": cfg.time_budget_sec,
            "seeds": list(cfg.seeds),
            "max_static_instances": cfg.max_static_instances,
            "max_online_instances": cfg.max_online_instances,
            "max_chained_instances": cfg.max_chained_instances,
            "events_per_instance": cfg.events_per_instance,
            "stnu_threshold": float(args.stnu_threshold),
        },
        "external_baselines_attempted": BASELINES,
        "external_baselines_integrated_and_run": BASELINES,
        "outputs": {
            "raw_static_rows": str(raw_dir / "external_baseline_static_rows.csv"),
            "raw_online_rows": str(raw_dir / "external_baseline_online_rows.csv"),
            "raw_chained_rows": str(raw_dir / "external_baseline_chained_rows.csv"),
            "table_external_baseline_static_main": str(tables_dir / "table_external_baseline_static_main.csv"),
            "table_external_baseline_online_main": str(tables_dir / "table_external_baseline_online_main.csv"),
            "table_external_baseline_harder_split": str(tables_dir / "table_external_baseline_harder_split.csv"),
            "table_external_baseline_chained": str(tables_dir / "table_external_baseline_chained.csv"),
            "table_external_baseline_master_summary": str(tables_dir / "table_external_baseline_master_summary.csv"),
            "reference_root": str(reference_root),
        },
    }
    write_json(out_root / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
