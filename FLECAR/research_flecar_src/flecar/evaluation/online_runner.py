from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from qcma.graph.reduced_graph import build_reduced_graph
from qcma.repair.event_simulator import (
    EVENT_TYPES,
    adaptive_scope_controller,
    apply_event_to_bans,
    edit_distance_by_subtask,
    frozen_subtasks_at_time,
    sample_trigger_times,
)
from qcma.repair.repair_cpsat import solve_repair_cpsat
from qcma.solvers.common import is_feasible
from qcma.solvers.cpsat_runner import solve_cpsat

from flecar.compiler import FESScoreWeights, compile_fes_candidates
from flecar.dispatch import check_dispatchability, dispatch_step
from flecar.envelope import synthesize_envelope
from flecar.evaluation.common_metrics import common_next_event_survivability, deterministic_rng, normalized_repair_burden
from flecar.repair import event_diagnostics, localize_controllability_region, run_car_local_resynthesis
from flecar.utils import read_json


@dataclass
class OnlineRunConfig:
    top_k: int
    density_scale: float
    density_seed: int
    time_budget_sec: float
    events_per_instance: int
    seed: int
    lambda_e: float = 4.0
    lambda_r: float = 1.0
    lambda_c: float = 2.0
    lambda_m: float = 1.5
    base_edit_penalty: float = 8.0


def _planner_only_proxy(reduced: dict[str, Any]) -> list[str]:
    nodes = reduced["nodes"]
    selected = []
    for _sid, cids in sorted(reduced["unique_groups"].items()):
        best = min(cids, key=lambda cid: (int(nodes[cid]["start"]), float(nodes[cid]["cost"]), str(cid)))
        selected.append(best)
    return selected


def _normalize_event_type(event_type: str) -> str:
    if event_type == "temporary_agent_unavailability":
        return "agent_unavailable"
    return str(event_type)


def _init_method_states(inst: dict[str, Any], cfg: OnlineRunConfig) -> dict[str, dict[str, Any]]:
    states: dict[str, dict[str, Any]] = {}

    reduced_exp = build_reduced_graph(
        inst,
        top_k=cfg.top_k,
        constraint_density_scale=cfg.density_scale,
        density_seed=cfg.density_seed,
        candidate_rank_mode="cost_start_duration",
        enable_dominated_pruning=True,
    )
    out_exp = solve_cpsat(reduced_exp, time_limit_sec=cfg.time_budget_sec)
    sel_exp = list(out_exp.get("selected_candidate_ids", []))

    reduced_bacb = build_reduced_graph(
        inst,
        top_k=cfg.top_k,
        constraint_density_scale=cfg.density_scale,
        density_seed=cfg.density_seed,
        candidate_rank_mode="bottleneck_diverse",
        enable_dominated_pruning=True,
    )
    out_bacb = solve_cpsat(reduced_bacb, time_limit_sec=cfg.time_budget_sec)
    sel_bacb = list(out_bacb.get("selected_candidate_ids", []))

    compiled_fes, _score_rows = compile_fes_candidates(inst, top_k=cfg.top_k, weights=FESScoreWeights())
    reduced_fes = build_reduced_graph(
        compiled_fes,
        top_k=100,
        constraint_density_scale=cfg.density_scale,
        density_seed=cfg.density_seed,
        candidate_rank_mode="cost_start_duration",
        enable_dominated_pruning=True,
    )
    out_fes = solve_cpsat(reduced_fes, time_limit_sec=cfg.time_budget_sec)
    sel_fes = list(out_fes.get("selected_candidate_ids", []))
    env_fes = synthesize_envelope(reduced_fes, sel_fes)

    sel_proxy = _planner_only_proxy(reduced_exp)

    states["planner_only_baseline"] = {
        "reduced": reduced_exp,
        "selected": sel_proxy,
        "envelope": None,
        "repair_mode": "none",
    }
    states["explicit_scheduling"] = {
        "reduced": reduced_exp,
        "selected": sel_exp,
        "envelope": None,
        "repair_mode": "none",
    }
    states["explicit_scheduling_bacb"] = {
        "reduced": reduced_bacb,
        "selected": sel_bacb,
        "envelope": None,
        "repair_mode": "none",
    }
    states["point_legacy_repair"] = {
        "reduced": reduced_exp,
        "selected": list(sel_exp),
        "envelope": None,
        "repair_mode": "legacy",
    }
    states["point_car_like_repair"] = {
        "reduced": reduced_exp,
        "selected": list(sel_exp),
        "envelope": None,
        "repair_mode": "point_car_like",
    }
    states["fes_no_car"] = {
        "reduced": reduced_fes,
        "selected": list(sel_fes),
        "envelope": dict(env_fes),
        "repair_mode": "none_fes",
    }
    states["fes_car"] = {
        "reduced": reduced_fes,
        "selected": list(sel_fes),
        "envelope": dict(env_fes),
        "repair_mode": "car",
    }
    states["fes_eawps"] = {
        "reduced": reduced_fes,
        "selected": list(sel_fes),
        "envelope": dict(env_fes),
        "repair_mode": "eawps",
    }
    return states


def run_online_block(
    instance_files: list[Path],
    cfg: OnlineRunConfig,
    *,
    forced_event_types: list[str] | None = None,
    adversarial_near_critical: bool = False,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for p in instance_files:
        inst = read_json(p)
        eid = str(inst["episode_id"])
        states = _init_method_states(inst, cfg)

        any_reduced = next(iter(states.values()))["reduced"]
        base_selected = next(iter(states.values()))["selected"]
        trigger_times = sample_trigger_times(any_reduced, base_selected, cfg.events_per_instance)

        if adversarial_near_critical:
            fes_env = states.get("fes_car", {}).get("envelope", {})
            windows = fes_env.get("windows", {})
            narrow = sorted(
                [(sid, float(w["u"]) - float(w["l"])) for sid, w in windows.items()],
                key=lambda x: x[1],
            )
            crit_sids = [sid for sid, _ in narrow[: max(1, cfg.events_per_instance)]]
            sid_to_time = {sid: int(float(windows[sid]["l"])) + 1 for sid in crit_sids if sid in windows}
            adv_times = list(sid_to_time.values())
            if adv_times:
                while len(adv_times) < cfg.events_per_instance:
                    adv_times.append(adv_times[-1] + 1)
                trigger_times = sorted(adv_times[: cfg.events_per_instance])

        for ev_idx, t in enumerate(trigger_times):
            if forced_event_types:
                event_type = str(forced_event_types[ev_idx % len(forced_event_types)])
            else:
                event_type = str(EVENT_TYPES[ev_idx % len(EVENT_TYPES)])
            event_for_sim = _normalize_event_type(event_type)

            for method, st in states.items():
                reduced = st["reduced"]
                prev_selected = list(st["selected"])
                env = st.get("envelope")

                event_rng = deterministic_rng("online_event", cfg.seed, eid, ev_idx, int(t), event_for_sim)
                bans = apply_event_to_bans(
                    reduced,
                    prev_selected,
                    trigger_time=int(t),
                    event_type=event_for_sim,
                    rng=event_rng,
                )
                bans_set = {str(x) for x in bans}

                repair_latency = 0.0
                edits = 0
                repair_regret = 0.0
                envelope_churn = 0.0
                repair_free_absorption = 0
                ctrl_preserve = 0
                repair_triggered = 0

                if st["repair_mode"] == "none":
                    new_selected = [cid for cid in prev_selected if str(cid) not in bans_set]

                elif st["repair_mode"] == "legacy":
                    frozen = frozen_subtasks_at_time(reduced, prev_selected, trigger_time=int(t))
                    t0 = perf_counter()
                    out = solve_repair_cpsat(
                        reduced,
                        prev_selected,
                        frozen_subtasks=frozen,
                        banned_candidates=bans_set,
                        edit_penalty=float(cfg.base_edit_penalty),
                        time_limit_sec=cfg.time_budget_sec,
                    )
                    repair_latency = perf_counter() - t0
                    new_selected = list(out.get("selected_candidate_ids", []))
                    edits = int(edit_distance_by_subtask(reduced, prev_selected, new_selected))
                    envelope_churn = float(edits) / float(max(1, len(reduced.get("unique_groups", {}))))
                    repair_triggered = 1

                elif st["repair_mode"] == "none_fes":
                    disp = dispatch_step(
                        reduced,
                        env,
                        prev_selected,
                        banned_candidates=bans_set,
                        current_time=int(t),
                    )
                    new_selected = list(disp["selection"])
                    repair_free_absorption = int(disp["dispatchable"])
                    ctrl_preserve = int(float(disp.get("controllability_margin", 0.0)) >= 0.0)
                    env = disp.get("effective_envelope", env)
                    st["envelope"] = env

                elif st["repair_mode"] == "point_car_like":
                    # Point schedule + CAR-like repair: rebuild a narrower on-the-fly envelope
                    # from the point schedule rather than relying on an offline FES envelope.
                    if env is None:
                        env = synthesize_envelope(
                            reduced,
                            prev_selected,
                            slack_expand_ratio=0.10,
                            min_window_pad=1,
                            optimize_grid=False,
                        )
                    diag = event_diagnostics(
                        reduced,
                        env,
                        prev_selected,
                        event_type=event_type,
                        trigger_time=int(t),
                        banned_candidates=bans_set,
                    )
                    if diag["dispatchable_after_event"]:
                        disp = dispatch_step(
                            reduced,
                            env,
                            prev_selected,
                            banned_candidates=bans_set,
                            current_time=int(t),
                        )
                        new_selected = list(disp["selection"])
                        repair_free_absorption = int(disp["dispatchable"])
                        ctrl_preserve = int(float(disp.get("controllability_margin", 0.0)) >= 0.0)
                        envelope_churn = 0.0
                        edits = 0
                        env = disp.get("effective_envelope", env)
                        st["envelope"] = env
                    else:
                        local = localize_controllability_region(
                            reduced,
                            diag,
                            radius=1,
                        )
                        t0 = perf_counter()
                        out = run_car_local_resynthesis(
                            reduced,
                            prev_selected,
                            trigger_time=int(t),
                            banned_candidates=bans_set,
                            localized_subtasks=local,
                            lambda_e=cfg.lambda_e,
                            lambda_r=cfg.lambda_r,
                            lambda_c=cfg.lambda_c,
                            lambda_m=cfg.lambda_m,
                            base_edit_penalty=cfg.base_edit_penalty,
                            diagnostics=diag,
                            time_budget_sec=cfg.time_budget_sec,
                            prev_envelope=env,
                        )
                        repair_latency = perf_counter() - t0
                        new_selected = list(out.get("selected_candidate_ids", []))
                        edits = int(out.get("car_meta", {}).get("edit_burden", 0))
                        envelope_churn = float(out.get("car_meta", {}).get("envelope_churn", 0.0))
                        repair_triggered = 1

                        if out.get("envelope") is not None:
                            env = out["envelope"]
                        else:
                            env = synthesize_envelope(
                                reduced,
                                new_selected,
                                slack_expand_ratio=0.10,
                                min_window_pad=1,
                                optimize_grid=False,
                            )
                        st["envelope"] = env
                        ctrl_preserve = int(float(env.get("metrics", {}).get("controllability_margin", 0.0)) >= 0.0)

                elif st["repair_mode"] == "car":
                    diag = event_diagnostics(
                        reduced,
                        env,
                        prev_selected,
                        event_type=event_type,
                        trigger_time=int(t),
                        banned_candidates=bans_set,
                    )
                    if diag["dispatchable_after_event"]:
                        disp = dispatch_step(
                            reduced,
                            env,
                            prev_selected,
                            banned_candidates=bans_set,
                            current_time=int(t),
                        )
                        new_selected = list(disp["selection"])
                        repair_free_absorption = int(disp["dispatchable"])
                        ctrl_preserve = int(float(disp.get("controllability_margin", 0.0)) >= 0.0)
                        envelope_churn = 0.0
                        edits = 0
                        env = disp.get("effective_envelope", env)
                        st["envelope"] = env
                    else:
                        local = localize_controllability_region(
                            reduced,
                            diag,
                            radius=1,
                        )
                        t0 = perf_counter()
                        out = run_car_local_resynthesis(
                            reduced,
                            prev_selected,
                            trigger_time=int(t),
                            banned_candidates=bans_set,
                            localized_subtasks=local,
                            lambda_e=cfg.lambda_e,
                            lambda_r=cfg.lambda_r,
                            lambda_c=cfg.lambda_c,
                            lambda_m=cfg.lambda_m,
                            base_edit_penalty=cfg.base_edit_penalty,
                            diagnostics=diag,
                            time_budget_sec=cfg.time_budget_sec,
                            prev_envelope=env,
                        )
                        repair_latency = perf_counter() - t0
                        new_selected = list(out.get("selected_candidate_ids", []))
                        edits = int(out.get("car_meta", {}).get("edit_burden", 0))
                        envelope_churn = float(out.get("car_meta", {}).get("envelope_churn", 0.0))
                        repair_triggered = 1

                        if out.get("envelope") is not None:
                            env = out["envelope"]
                        else:
                            env = synthesize_envelope(reduced, new_selected)
                        st["envelope"] = env
                        ctrl_preserve = int(float(env.get("metrics", {}).get("controllability_margin", 0.0)) >= 0.0)

                elif st["repair_mode"] == "eawps":
                    ctrl = adaptive_scope_controller(
                        reduced,
                        prev_selected,
                        trigger_time=int(t),
                        banned_candidates=bans_set,
                        event_type=event_for_sim,
                        base_edit_penalty=float(cfg.base_edit_penalty),
                        default_radius=1,
                    )
                    t0 = perf_counter()
                    out = solve_repair_cpsat(
                        reduced,
                        prev_selected,
                        frozen_subtasks=set(ctrl["frozen_subtasks"]),
                        banned_candidates=bans_set,
                        editable_subtasks=set(ctrl["editable_subtasks"]),
                        edit_penalty=float(ctrl["edit_penalty"]),
                        time_limit_sec=cfg.time_budget_sec,
                    )
                    repair_latency = perf_counter() - t0
                    new_selected = list(out.get("selected_candidate_ids", []))
                    edits = int(edit_distance_by_subtask(reduced, prev_selected, new_selected))
                    envelope_churn = float(edits) / float(max(1, len(reduced.get("unique_groups", {}))))
                    env = synthesize_envelope(reduced, new_selected)
                    st["envelope"] = env
                    ctrl_preserve = int(float(env.get("metrics", {}).get("controllability_margin", 0.0)) >= 0.0)
                    repair_triggered = 1

                else:
                    new_selected = [cid for cid in prev_selected if str(cid) not in bans_set]

                feasible = int(is_feasible(reduced, new_selected))
                n_subtasks = len(reduced.get("unique_groups", {}))
                frozen_now = frozen_subtasks_at_time(reduced, new_selected, trigger_time=int(t))
                cumulative_completion = min(1.0, (len(frozen_now) + feasible) / float(max(1, n_subtasks)))
                burden = normalized_repair_burden(
                    reduced,
                    prev_selected,
                    new_selected,
                    feasible=bool(feasible),
                    repair_latency=float(repair_latency),
                    repair_triggered=int(repair_triggered),
                    time_budget_sec=float(cfg.time_budget_sec),
                )
                repair_regret = float(burden["repair_burden"])
                future_rng = deterministic_rng("future_probe", cfg.seed, eid, ev_idx, int(t), event_for_sim)
                future_feas = common_next_event_survivability(
                    reduced,
                    new_selected,
                    trigger_time=int(t),
                    rng=future_rng,
                )

                env_now = st.get("envelope")
                if env_now is None and method.startswith("fes"):
                    env_now = synthesize_envelope(reduced, new_selected)
                    st["envelope"] = env_now

                ctrl_margin = float(env_now.get("metrics", {}).get("controllability_margin", 0.0)) if env_now else 0.0
                if method.startswith("fes") and ctrl_preserve == 0:
                    ctrl_preserve = int(ctrl_margin >= 0.0)

                rows.append(
                    {
                        "episode_id": eid,
                        "event_idx": int(ev_idx),
                        "event_type": str(event_type),
                        "trigger_time": int(t),
                        "method": method,
                        "post_event_feasible": int(feasible),
                        "cumulative_completion": float(cumulative_completion),
                        "terminal_success_proxy": int(feasible if ev_idx == len(trigger_times) - 1 else 0),
                        "terminal_completion_proxy": float(cumulative_completion if ev_idx == len(trigger_times) - 1 else 0.0),
                        "repair_free_absorption": int(repair_free_absorption),
                        "controllability_preserved": int(ctrl_preserve),
                        "future_feasibility_after_next_event": int(future_feas),
                        "one_step_survivability": int(future_feas),
                        "repair_regret": float(repair_regret),
                        "repair_burden": float(repair_regret),
                        "envelope_churn": float(envelope_churn),
                        "edit_burden": int(edits),
                        "repair_latency": float(repair_latency),
                        "controllability_margin": float(ctrl_margin),
                        "repair_triggered": int(repair_triggered),
                    }
                )

                st["selected"] = list(new_selected)

    return rows


def summarize_online(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_method: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_method.setdefault(str(r["method"]), []).append(r)

    out: list[dict[str, Any]] = []
    for m, vals in sorted(by_method.items()):
        n = len(vals)
        by_ep: dict[str, list[dict[str, Any]]] = {}
        for v in vals:
            by_ep.setdefault(str(v["episode_id"]), []).append(v)
        terminals = [sorted(vs, key=lambda x: int(x["event_idx"]))[-1] for vs in by_ep.values()]

        out.append(
            {
                "method": m,
                "n_events": n,
                "n_episodes": len(by_ep),
                "post_event_feasibility_rate": sum(int(v["post_event_feasible"]) for v in vals) / max(1, n),
                "mean_cumulative_completion": sum(float(v["cumulative_completion"]) for v in vals) / max(1, n),
                "terminal_success_rate": sum(int(v["post_event_feasible"]) for v in terminals) / max(1, len(terminals)),
                "terminal_completion_mean": sum(float(v["cumulative_completion"]) for v in terminals) / max(1, len(terminals)),
                "repair_free_absorption_rate": sum(int(v["repair_free_absorption"]) for v in vals) / max(1, n),
                "controllability_preservation_rate": sum(int(v["controllability_preserved"]) for v in vals) / max(1, n),
                "future_feasibility_after_next_event": sum(int(v["future_feasibility_after_next_event"]) for v in vals) / max(1, n),
                "mean_repair_regret": sum(float(v["repair_regret"]) for v in vals) / max(1, n),
                "mean_repair_burden": sum(float(v.get("repair_burden", v["repair_regret"])) for v in vals) / max(1, n),
                "mean_envelope_churn": sum(float(v["envelope_churn"]) for v in vals) / max(1, n),
                "mean_edit_burden": sum(float(v["edit_burden"]) for v in vals) / max(1, n),
                "mean_repair_latency": sum(float(v["repair_latency"]) for v in vals) / max(1, n),
                "repair_trigger_rate": sum(int(v.get("repair_triggered", 0)) for v in vals) / max(1, n),
            }
        )
    return out
