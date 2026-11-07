from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC_Q = ROOT / "src"
SRC_F = ROOT / "research_flecar" / "src"
for p in [SRC_Q, SRC_F]:
    if str(p) not in sys.path:
        sys.path.append(str(p))

from flecar.evaluation import StaticRunConfig, run_static_block, summarize_static  # noqa: E402
from flecar.utils import list_instance_files, write_csv, write_json  # noqa: E402


def _aggregate_summary(all_rows: list[dict[str, Any]], key_fields: list[str]) -> list[dict[str, Any]]:
    by: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for r in all_rows:
        k = tuple(r[kf] for kf in key_fields)
        by.setdefault(k, []).append(r)

    out: list[dict[str, Any]] = []
    for k, vals in sorted(by.items()):
        row = {kf: k[i] for i, kf in enumerate(key_fields)}
        for metric in [
            "feasible_rate",
            "mean_violations_total",
            "mean_runtime_sec",
            "mean_objective_feasible_only",
            "mean_envelope_flexibility_volume",
            "mean_critical_slack_margin",
            "mean_dispatchability_success",
            "mean_absorbable_disturbance_radius",
        ]:
            row[metric] = sum(float(v[metric]) for v in vals) / max(1, len(vals))
        row["n_instances"] = int(sum(int(v.get("n_instances", 0)) for v in vals) / max(1, len(vals)))
        out.append(row)
    return out


def _quantile(vals: list[float], q: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    idx = int(round((len(s) - 1) * max(0.0, min(1.0, q))))
    return float(s[idx])


def _safe_ratio(delta: float, base: float) -> float | None:
    if abs(float(base)) < 1e-9:
        return None
    return float(delta / abs(float(base)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances-dir", type=Path, default=Path("research_flecar/data_build/flecar_instances_v1"))
    ap.add_argument("--out-dir", type=Path, default=Path("research_flecar/outputs/static"))
    ap.add_argument("--seeds", type=str, default="7,11,19,23,29")
    ap.add_argument("--max-instances", type=int, default=300)
    # Canonical claim defaults (aligned with flecar_static_main config).
    ap.add_argument("--top-k", type=int, default=2)
    ap.add_argument("--density-scale", type=float, default=1.5)
    ap.add_argument("--time-budget-sec", type=float, default=0.5)
    args = ap.parse_args()

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    files = list_instance_files(ROOT / args.instances_dir)
    files = files[: int(args.max_instances)]
    seeds = [int(x.strip()) for x in str(args.seeds).split(",") if x.strip()]

    all_static_rows: list[dict[str, Any]] = []
    all_summary_rows: list[dict[str, Any]] = []
    all_comp_rows: list[dict[str, Any]] = []
    all_paired_rows: list[dict[str, Any]] = []
    all_scores: list[dict[str, Any]] = []

    for seed in seeds:
        cfg = StaticRunConfig(
            top_k=int(args.top_k),
            density_scale=float(args.density_scale),
            density_seed=seed,
            time_budget_sec=float(args.time_budget_sec),
            seed=seed,
        )
        rows, paired, scores = run_static_block(files, cfg)
        summary, comps = summarize_static(rows, paired, seed=seed)

        for r in rows:
            all_static_rows.append({**r, "seed": seed})
        for r in paired:
            all_paired_rows.append({**r, "seed": seed})
        for r in summary:
            all_summary_rows.append({**r, "seed": seed})
        for r in comps:
            all_comp_rows.append({**r, "seed": seed})
        for r in scores:
            all_scores.append({**r, "seed": seed})

    summary_overall = _aggregate_summary(all_summary_rows, ["method"])

    # Main static block groups G1-G5 mapping.
    main_groups = {
        "G1_planner_only": "planner_only_baseline",
        "G2_explicit": "explicit_scheduling",
        "G3_explicit_bacb": "explicit_scheduling_bacb",
        "G4_fes": "fes",
        "G5_fes_static_only": "fes_static_only",
    }
    idx = {str(r["method"]): r for r in summary_overall}
    static_main_rows = []
    for gid, m in main_groups.items():
        if m not in idx:
            continue
        row = dict(idx[m])
        row["group"] = gid
        static_main_rows.append(row)

    # Harder-slice summaries.
    n_subtasks_vals = [float(r.get("n_subtasks", 0)) for r in all_static_rows]
    n_conflict_vals = [float(r.get("n_conflict_edges", 0)) for r in all_static_rows]
    density_vals = [float(r.get("candidate_density", 0.0)) for r in all_static_rows]
    thr_subtasks = 2.0
    thr_conflict = max(5.0, _quantile(n_conflict_vals, 0.85))
    thr_density = _quantile(density_vals, 0.80)

    harder_rows = []
    for method in sorted({str(r["method"]) for r in all_static_rows}):
        vals = [r for r in all_static_rows if str(r["method"]) == method]
        larger = [r for r in vals if float(r.get("n_subtasks", 0)) >= thr_subtasks]
        high_conf = [r for r in vals if float(r.get("n_conflict_edges", 0)) >= thr_conflict]
        high_density = [r for r in vals if float(r.get("candidate_density", 0.0)) >= thr_density]
        for tag, xs in [
            ("larger_graph", larger),
            ("higher_conflict", high_conf),
            ("higher_candidate_density", high_density),
        ]:
            if not xs:
                continue
            harder_rows.append(
                {
                    "method": method,
                    "slice": tag,
                    "n_rows": len(xs),
                    "feasible_rate": sum(int(x["feasible"]) for x in xs) / max(1, len(xs)),
                    "mean_violations_total": sum(float(x["violations_total"]) for x in xs) / max(1, len(xs)),
                    "mean_runtime_sec": sum(float(x["runtime_sec"]) for x in xs) / max(1, len(xs)),
                    "mean_flexibility_volume": sum(float(x["envelope_flexibility_volume"]) for x in xs) / max(1, len(xs)),
                    "mean_critical_slack_margin": sum(float(x["critical_slack_margin"]) for x in xs) / max(1, len(xs)),
                    "mean_dispatchability_success": sum(float(x["dispatchability_success"]) for x in xs) / max(1, len(xs)),
                    "mean_absorbable_disturbance_radius": sum(float(x["absorbable_disturbance_radius"]) for x in xs) / max(1, len(xs)),
                }
            )

    # Threshold-focused checks.
    fes = idx.get("fes")
    point_methods = ["explicit_scheduling", "explicit_scheduling_bacb"]
    point_pool = [idx[m] for m in point_methods if m in idx]
    strong = max(point_pool, key=lambda r: float(r.get("feasible_rate", 0.0))) if point_pool else None
    strong_name = str(strong.get("method", "")) if strong else ""
    threshold_table = []
    if fes and strong:
        feasible_delta = float(fes["feasible_rate"]) - float(strong["feasible_rate"])
        dispatch_gain = float(fes["mean_dispatchability_success"]) - float(strong["mean_dispatchability_success"])
        flex_delta = float(fes["mean_envelope_flexibility_volume"]) - float(strong["mean_envelope_flexibility_volume"])
        csm_delta = float(fes["mean_critical_slack_margin"]) - float(strong["mean_critical_slack_margin"])
        adr_delta = float(fes["mean_absorbable_disturbance_radius"]) - float(strong["mean_absorbable_disturbance_radius"])

        flex_ratio = _safe_ratio(flex_delta, float(strong["mean_envelope_flexibility_volume"]))
        csm_ratio = _safe_ratio(csm_delta, float(strong["mean_critical_slack_margin"]))
        adr_ratio = _safe_ratio(adr_delta, float(strong["mean_absorbable_disturbance_radius"]))

        flex_pass = (float(flex_ratio) >= 0.15) if (flex_ratio is not None) else (float(flex_delta) >= 0.15)
        csm_pass = (float(csm_ratio) >= 0.15) if (csm_ratio is not None) else (float(csm_delta) >= 0.15)
        adr_pass = (float(adr_ratio) >= 0.20) if (adr_ratio is not None) else (float(adr_delta) >= 0.20)

        harder_idx: dict[tuple[str, str], dict[str, Any]] = {}
        for r in harder_rows:
            harder_idx[(str(r["method"]), str(r["slice"]))] = r
        larger_delta = None
        conflict_delta = None
        if ("fes", "larger_graph") in harder_idx and (strong_name, "larger_graph") in harder_idx:
            larger_delta = float(harder_idx[("fes", "larger_graph")]["feasible_rate"]) - float(harder_idx[(strong_name, "larger_graph")]["feasible_rate"])
        if ("fes", "higher_conflict") in harder_idx and (strong_name, "higher_conflict") in harder_idx:
            conflict_delta = float(harder_idx[("fes", "higher_conflict")]["feasible_rate"]) - float(harder_idx[(strong_name, "higher_conflict")]["feasible_rate"])

        threshold_table.append(
            {
                "comparison": f"fes_vs_{strong_name}",
                "reference_method": strong_name,
                "overall_feasible_delta": feasible_delta,
                "larger_graph_feasible_delta": larger_delta if larger_delta is not None else "",
                "higher_conflict_feasible_delta": conflict_delta if conflict_delta is not None else "",
                "dispatchability_success_gain": dispatch_gain,
                "flexibility_volume_gain_delta": flex_delta,
                "critical_slack_margin_gain_delta": csm_delta,
                "absorbable_disturbance_radius_gain_delta": adr_delta,
                "flexibility_volume_gain_ratio": float(flex_ratio) if flex_ratio is not None else "",
                "critical_slack_margin_gain_ratio": float(csm_ratio) if csm_ratio is not None else "",
                "absorbable_disturbance_radius_gain_ratio": float(adr_ratio) if adr_ratio is not None else "",
                "flexibility_volume_ratio_defined": int(flex_ratio is not None),
                "critical_slack_margin_ratio_defined": int(csm_ratio is not None),
                "absorbable_disturbance_radius_ratio_defined": int(adr_ratio is not None),
                "pass_overall_feasible_delta_ge_0_10": int(feasible_delta >= 0.10),
                "pass_harder_delta_larger_than_overall": int(
                    (larger_delta is not None and larger_delta >= feasible_delta)
                    or (conflict_delta is not None and conflict_delta >= feasible_delta)
                ),
                "pass_dispatchability_gain_ge_0_08": int(dispatch_gain >= 0.08),
                "pass_flex_or_csm_gain_ge_0_15": int(bool(flex_pass and csm_pass)),
                "pass_adr_gain_ge_0_20": int(bool(adr_pass)),
            }
        )

    write_csv(out_dir / "static_rows.csv", all_static_rows)
    write_csv(out_dir / "static_paired_rows.csv", all_paired_rows)
    write_csv(out_dir / "fes_candidate_scores.csv", all_scores)
    write_csv(out_dir / "static_summary_by_seed.csv", all_summary_rows)
    write_csv(out_dir / "static_summary_overall.csv", summary_overall)
    write_csv(out_dir / "static_paired_comparisons_by_seed.csv", all_comp_rows)
    write_csv(out_dir / "static_harder_slice_summary.csv", harder_rows)
    write_csv(out_dir / "static_main_comparison.csv", static_main_rows)
    write_csv(out_dir / "static_threshold_table.csv", threshold_table)

    manifest = {
        "run_id": f"flecar_static_{len(seeds)}seeds_{len(files)}instances",
        "n_instances": len(files),
        "seeds": seeds,
        "top_k": int(args.top_k),
        "density_scale": float(args.density_scale),
        "time_budget_sec": float(args.time_budget_sec),
        "outputs": {
            "static_rows": str(out_dir / "static_rows.csv"),
            "summary_overall": str(out_dir / "static_summary_overall.csv"),
            "harder_slice_summary": str(out_dir / "static_harder_slice_summary.csv"),
            "main_comparison": str(out_dir / "static_main_comparison.csv"),
            "threshold_table": str(out_dir / "static_threshold_table.csv"),
        },
    }
    write_json(out_dir / "manifest.json", manifest)
    write_json(ROOT / "research_flecar/outputs/latest_static_manifest.json", manifest)
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
