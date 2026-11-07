from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qcma.eval.metrics_static import evaluate_static_solution
from qcma.graph.reduced_graph import build_reduced_graph
from qcma.solvers.cpsat_runner import solve_cpsat

from flecar.analysis.stats import paired_summary
from flecar.compiler import FESScoreWeights, compile_bacb_plus_candidates, compile_fes_candidates
from flecar.dispatch import absorbable_disturbance_radius, check_dispatchability
from flecar.envelope import synthesize_envelope
from flecar.utils import read_json


@dataclass
class StaticRunConfig:
    top_k: int
    density_scale: float
    density_seed: int
    time_budget_sec: float
    seed: int
    lambda_obj: float = 1.0
    lambda_flex: float = 0.4
    lambda_risk: float = 0.35


def _planner_only_proxy(reduced: dict[str, Any]) -> dict[str, Any]:
    nodes = reduced["nodes"]
    selected = []
    for _sid, cids in sorted(reduced["unique_groups"].items()):
        best = min(cids, key=lambda cid: (int(nodes[cid]["start"]), float(nodes[cid]["cost"]), str(cid)))
        selected.append(best)
    return {
        "solver": "planner_only_proxy",
        "selected_candidate_ids": selected,
        "runtime_sec": 0.0,
        "solver_status": "PROXY",
    }


def _explicit_schedule(inst: dict[str, Any], cfg: StaticRunConfig, *, rank_mode: str) -> tuple[dict[str, Any], dict[str, Any]]:
    reduced = build_reduced_graph(
        inst,
        top_k=cfg.top_k,
        constraint_density_scale=cfg.density_scale,
        density_seed=cfg.density_seed,
        candidate_rank_mode=rank_mode,
        enable_dominated_pruning=True,
    )
    out = solve_cpsat(reduced, time_limit_sec=cfg.time_budget_sec)
    return reduced, out


def _fes_schedule(
    inst: dict[str, Any],
    cfg: StaticRunConfig,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    weights = FESScoreWeights()
    compiled_inst, score_rows = compile_fes_candidates(inst, top_k=cfg.top_k, weights=weights)
    reduced = build_reduced_graph(
        compiled_inst,
        top_k=100,
        constraint_density_scale=cfg.density_scale,
        density_seed=cfg.density_seed,
        candidate_rank_mode="cost_start_duration",
        enable_dominated_pruning=True,
    )
    out = solve_cpsat(reduced, time_limit_sec=cfg.time_budget_sec)
    envelope = synthesize_envelope(
        reduced,
        out.get("selected_candidate_ids", []),
        lambda_obj=cfg.lambda_obj,
        lambda_flex=cfg.lambda_flex,
        lambda_risk=cfg.lambda_risk,
    )
    return reduced, out, envelope, score_rows


def _fes_static_only(inst: dict[str, Any], cfg: StaticRunConfig) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    compiled_inst, _ = compile_fes_candidates(inst, top_k=cfg.top_k, weights=FESScoreWeights())
    reduced = build_reduced_graph(
        compiled_inst,
        top_k=100,
        constraint_density_scale=cfg.density_scale,
        density_seed=cfg.density_seed,
        candidate_rank_mode="cost_start_duration",
        enable_dominated_pruning=True,
    )
    out = solve_cpsat(reduced, time_limit_sec=cfg.time_budget_sec)
    envelope = synthesize_envelope(
        reduced,
        out.get("selected_candidate_ids", []),
        lambda_obj=cfg.lambda_obj,
        lambda_flex=cfg.lambda_flex,
        lambda_risk=cfg.lambda_risk,
    )
    return reduced, out, envelope


def run_static_block(instance_files: list[Path], cfg: StaticRunConfig) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []

    for p in instance_files:
        inst = read_json(p)
        eid = str(inst["episode_id"])

        r_proxy = build_reduced_graph(
            inst,
            top_k=cfg.top_k,
            constraint_density_scale=cfg.density_scale,
            density_seed=cfg.density_seed,
            candidate_rank_mode="cost_start_duration",
            enable_dominated_pruning=True,
        )
        o_proxy = _planner_only_proxy(r_proxy)
        m_proxy = evaluate_static_solution(r_proxy, o_proxy)

        r_exp, o_exp = _explicit_schedule(inst, cfg, rank_mode="cost_start_duration")
        m_exp = evaluate_static_solution(r_exp, o_exp)

        r_bacb, o_bacb = _explicit_schedule(inst, cfg, rank_mode="bottleneck_diverse")
        m_bacb = evaluate_static_solution(r_bacb, o_bacb)

        bacb_plus_inst = compile_bacb_plus_candidates(inst, top_k=cfg.top_k)
        r_bacbp = build_reduced_graph(
            bacb_plus_inst,
            top_k=100,
            constraint_density_scale=cfg.density_scale,
            density_seed=cfg.density_seed,
            candidate_rank_mode="cost_start_duration",
            enable_dominated_pruning=True,
        )
        o_bacbp = solve_cpsat(r_bacbp, time_limit_sec=cfg.time_budget_sec)
        m_bacbp = evaluate_static_solution(r_bacbp, o_bacbp)

        r_fes, o_fes, env_fes, srows = _fes_schedule(inst, cfg)
        m_fes = evaluate_static_solution(r_fes, o_fes)

        r_fes_static, o_fes_static, env_static = _fes_static_only(inst, cfg)
        m_fes_static = evaluate_static_solution(r_fes_static, o_fes_static)

        metrics_map = {
            "planner_only_baseline": (m_proxy, None, r_proxy),
            "explicit_scheduling": (m_exp, None, r_exp),
            "explicit_scheduling_bacb": (m_bacb, None, r_bacb),
            "explicit_scheduling_bacb_plus": (m_bacbp, None, r_bacbp),
            "fes": (m_fes, env_fes, r_fes),
            "fes_static_only": (m_fes_static, env_static, r_fes_static),
        }

        for method, (m, env, red) in metrics_map.items():
            em = env.get("metrics", {}) if env else {}
            disp = check_dispatchability(env) if env else {"dispatchable": False}
            adr = absorbable_disturbance_radius(env) if env else 0.0

            rows.append(
                {
                    "episode_id": eid,
                    "method": method,
                    "feasible": int(bool(m["feasible"])),
                    "violations_total": int(m["violations_total"]),
                    "runtime_sec": float(m["runtime_sec"]),
                    "objective": float(m["objective"]),
                    "envelope_flexibility_volume": float(em.get("flexibility_volume", 0.0)),
                    "critical_slack_margin": float(em.get("critical_slack_margin", 0.0)),
                    "sync_tolerance": float(em.get("sync_tolerance", 0.0)),
                    "resource_free_buffer": float(em.get("resource_free_buffer", 0.0)),
                    "dispatchability_success": int(bool(disp.get("dispatchable", False))),
                    "controllability_margin": float(em.get("controllability_margin", 0.0)),
                    "absorbable_disturbance_radius": float(adr),
                    "envelope_objective": float(env.get("objective", {}).get("value", 0.0)) if env else 0.0,
                    "n_subtasks": int(len(red.get("unique_groups", {}))),
                    "n_conflict_edges": int(len(red.get("conflict_edges", []))),
                    "candidate_density": float(len(red.get("nodes", {})) / max(1, len(red.get("unique_groups", {})))),
                    "solver_status": str(m.get("solver_meta", {}).get("solver_status", "")),
                }
            )

        for sr in srows:
            score_rows.append({"episode_id": eid, **sr})

        paired_rows.append(
            {
                "episode_id": eid,
                "proxy_feasible": int(bool(m_proxy["feasible"])),
                "explicit_feasible": int(bool(m_exp["feasible"])),
                "bacb_feasible": int(bool(m_bacb["feasible"])),
                "fes_feasible": int(bool(m_fes["feasible"])),
                "fes_static_only_feasible": int(bool(m_fes_static["feasible"])),
                "fes_dispatchability": int(bool(check_dispatchability(env_fes).get("dispatchable", False))),
                "fes_ctrl_margin": float(env_fes.get("metrics", {}).get("controllability_margin", 0.0)),
                "proxy_violations": int(m_proxy["violations_total"]),
                "explicit_violations": int(m_exp["violations_total"]),
                "bacb_violations": int(m_bacb["violations_total"]),
                "fes_violations": int(m_fes["violations_total"]),
                "proxy_runtime": float(m_proxy["runtime_sec"]),
                "explicit_runtime": float(m_exp["runtime_sec"]),
                "bacb_runtime": float(m_bacb["runtime_sec"]),
                "fes_runtime": float(m_fes["runtime_sec"]),
                "proxy_objective": float(m_proxy["objective"]),
                "explicit_objective": float(m_exp["objective"]),
                "bacb_objective": float(m_bacb["objective"]),
                "fes_objective": float(m_fes["objective"]),
                "fes_flex": float(env_fes.get("metrics", {}).get("flexibility_volume", 0.0)),
                "fes_csm": float(env_fes.get("metrics", {}).get("critical_slack_margin", 0.0)),
                "fes_dispatchable": int(bool(check_dispatchability(env_fes).get("dispatchable", False))),
                "fes_adr": float(absorbable_disturbance_radius(env_fes)),
            }
        )

    return rows, paired_rows, score_rows


def summarize_static(rows: list[dict[str, Any]], paired_rows: list[dict[str, Any]], seed: int = 7) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_method: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_method.setdefault(str(r["method"]), []).append(r)

    summary: list[dict[str, Any]] = []
    for m, vals in sorted(by_method.items()):
        n = len(vals)
        summary.append(
            {
                "method": m,
                "n_instances": n,
                "feasible_rate": sum(int(v["feasible"]) for v in vals) / max(1, n),
                "mean_violations_total": sum(float(v["violations_total"]) for v in vals) / max(1, n),
                "mean_runtime_sec": sum(float(v["runtime_sec"]) for v in vals) / max(1, n),
                "mean_objective_feasible_only": (
                    sum(float(v["objective"]) for v in vals if int(v["feasible"]) == 1)
                    / max(1, sum(1 for v in vals if int(v["feasible"]) == 1))
                ),
                "mean_envelope_flexibility_volume": sum(float(v["envelope_flexibility_volume"]) for v in vals) / max(1, n),
                "mean_critical_slack_margin": sum(float(v["critical_slack_margin"]) for v in vals) / max(1, n),
                "mean_dispatchability_success": sum(float(v["dispatchability_success"]) for v in vals) / max(1, n),
                "mean_absorbable_disturbance_radius": sum(float(v["absorbable_disturbance_radius"]) for v in vals) / max(1, n),
            }
        )

    comparisons: list[dict[str, Any]] = []
    for sched_key, base_key, name in [
        ("fes_feasible", "explicit_feasible", "fes_vs_explicit"),
        ("fes_feasible", "bacb_feasible", "fes_vs_bacb"),
        ("fes_feasible", "proxy_feasible", "fes_vs_proxy"),
    ]:
        ps = paired_summary(paired_rows, sched_key=sched_key, base_key=base_key, metric_key=name, seed=seed)
        comparisons.append({"comparison": name, **ps})

    return summary, comparisons
