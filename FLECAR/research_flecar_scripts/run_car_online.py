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

from flecar.evaluation import OnlineRunConfig, run_online_block, summarize_online  # noqa: E402
from flecar.utils import list_instance_files, write_csv, write_json  # noqa: E402


def _to_index(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(r["method"]): r for r in rows}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances-dir", type=Path, default=Path("research_flecar/data_build/flecar_instances_v1"))
    ap.add_argument("--out-dir", type=Path, default=Path("research_flecar/outputs/online"))
    ap.add_argument("--seeds", type=str, default="7,11,19,23,29")
    ap.add_argument("--max-instances", type=int, default=50)
    ap.add_argument("--events-per-instance", type=int, default=4)
    # Canonical claim defaults (aligned with flecar_online_main config).
    ap.add_argument("--top-k", type=int, default=2)
    ap.add_argument("--density-scale", type=float, default=1.5)
    ap.add_argument("--time-budget-sec", type=float, default=0.5)
    args = ap.parse_args()

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    files = list_instance_files(ROOT / args.instances_dir)[: int(args.max_instances)]
    seeds = [int(x.strip()) for x in str(args.seeds).split(",") if x.strip()]

    all_rows: list[dict[str, Any]] = []
    all_summary: list[dict[str, Any]] = []

    for seed in seeds:
        cfg = OnlineRunConfig(
            top_k=int(args.top_k),
            density_scale=float(args.density_scale),
            density_seed=seed,
            time_budget_sec=float(args.time_budget_sec),
            events_per_instance=int(args.events_per_instance),
            seed=seed,
        )
        rows = run_online_block(files, cfg)
        summary = summarize_online(rows)
        for r in rows:
            all_rows.append({**r, "seed": seed})
        for r in summary:
            all_summary.append({**r, "seed": seed})

    by_method: dict[str, list[dict[str, Any]]] = {}
    for r in all_summary:
        by_method.setdefault(str(r["method"]), []).append(r)

    overall: list[dict[str, Any]] = []
    for m, vals in sorted(by_method.items()):
        row = {
            "method": m,
            "n_seed_rows": len(vals),
        }
        for k in [
            "n_events",
            "n_episodes",
            "post_event_feasibility_rate",
            "mean_cumulative_completion",
            "terminal_success_rate",
            "terminal_completion_mean",
            "repair_free_absorption_rate",
            "controllability_preservation_rate",
            "future_feasibility_after_next_event",
            "mean_repair_regret",
            "mean_repair_burden",
            "mean_envelope_churn",
            "mean_edit_burden",
            "mean_repair_latency",
            "repair_trigger_rate",
        ]:
            row[k] = sum(float(v[k]) for v in vals) / max(1, len(vals))
        overall.append(row)

    # Block-2 main groups.
    main_groups = {
        "G1_planner_only": "planner_only_baseline",
        "G2_explicit": "explicit_scheduling",
        "G3_explicit_bacb": "explicit_scheduling_bacb",
        "G4_point_legacy_repair": "point_legacy_repair",
        "G5_fes_no_car": "fes_no_car",
        "G6_fes_car": "fes_car",
        "G7_fes_eawps": "fes_eawps",
    }
    idx = _to_index(overall)
    main_rows = []
    for gid, m in main_groups.items():
        if m not in idx:
            continue
        main_rows.append({"group": gid, **idx[m]})

    gains = []

    def add_gain(name: str, a: str, b: str) -> None:
        if a not in idx or b not in idx:
            return
        aa = idx[a]
        bb = idx[b]
        gains.append(
            {
                "comparison": name,
                "PostEventFeasibilityGain": float(aa["post_event_feasibility_rate"]) - float(bb["post_event_feasibility_rate"]),
                "CumulativeCompletionGain": float(aa["mean_cumulative_completion"]) - float(bb["mean_cumulative_completion"]),
                "TerminalSuccessGain": float(aa["terminal_success_rate"]) - float(bb["terminal_success_rate"]),
                "TerminalCompletionGain": float(aa["terminal_completion_mean"]) - float(bb["terminal_completion_mean"]),
                "RepairFreeAbsorptionRateGain": float(aa["repair_free_absorption_rate"]) - float(bb["repair_free_absorption_rate"]),
                "ControllabilityPreservationRateGain": float(aa["controllability_preservation_rate"]) - float(bb["controllability_preservation_rate"]),
                "FutureFeasibilityAfterNextEventGain": float(aa["future_feasibility_after_next_event"]) - float(bb["future_feasibility_after_next_event"]),
                "RepairRegretDelta": float(aa["mean_repair_regret"]) - float(bb["mean_repair_regret"]),
                "RepairBurdenDelta": float(aa.get("mean_repair_burden", aa["mean_repair_regret"])) - float(bb.get("mean_repair_burden", bb["mean_repair_regret"])),
                "EnvelopeChurnDelta": float(aa["mean_envelope_churn"]) - float(bb["mean_envelope_churn"]),
                "EditBurdenDelta": float(aa["mean_edit_burden"]) - float(bb["mean_edit_burden"]),
                "RepairLatencyDelta": float(aa["mean_repair_latency"]) - float(bb["mean_repair_latency"]),
            }
        )

    add_gain("point_legacy_repair_vs_explicit", "point_legacy_repair", "explicit_scheduling")
    add_gain("fes_car_vs_fes_no_car", "fes_car", "fes_no_car")
    add_gain("fes_car_vs_fes_eawps", "fes_car", "fes_eawps")
    add_gain("fes_car_vs_point_legacy_repair", "fes_car", "point_legacy_repair")

    # Online gate check (2 of 3 core metrics in this block, chained gain checked in chained block).
    gate_rows = []
    if "fes_car" in idx and "point_legacy_repair" in idx:
        a = idx["fes_car"]
        b = idx["point_legacy_repair"]
        abs_gain = float(a["repair_free_absorption_rate"]) - float(b["repair_free_absorption_rate"])
        future_gain = float(a["future_feasibility_after_next_event"]) - float(b["future_feasibility_after_next_event"])

        burden_improve = (float(b.get("mean_repair_burden", b["mean_repair_regret"])) - float(a.get("mean_repair_burden", a["mean_repair_regret"]))) / max(1e-6, abs(float(b.get("mean_repair_burden", b["mean_repair_regret"]))) + 1e-6)
        churn_improve = (float(b["mean_envelope_churn"]) - float(a["mean_envelope_churn"])) / max(1e-6, abs(float(b["mean_envelope_churn"])) + 1e-6)
        edit_improve = (float(b["mean_edit_burden"]) - float(a["mean_edit_burden"])) / max(1e-6, abs(float(b["mean_edit_burden"])) + 1e-6)
        burden_pass = max(burden_improve, churn_improve, edit_improve) >= 0.15

        pass_flags = [
            int(abs_gain >= 0.10),
            int(future_gain >= 0.08),
            int(burden_pass),
        ]
        gate_rows.append(
            {
                "comparison": "fes_car_vs_point_legacy_repair",
                "repair_free_absorption_gain": abs_gain,
                "future_feasibility_gain": future_gain,
                "repair_burden_improvement_ratio": burden_improve,
                "churn_improvement_ratio": churn_improve,
                "edit_improvement_ratio": edit_improve,
                "pass_absorption_gain_ge_0_10": int(abs_gain >= 0.10),
                "pass_future_feas_gain_ge_0_08": int(future_gain >= 0.08),
                "pass_burden_drop_ge_0_15": int(burden_pass),
                "pass_online_gate_2_of_3": int(sum(pass_flags) >= 2),
            }
        )

    write_csv(out_dir / "online_event_rows.csv", all_rows)
    write_csv(out_dir / "online_summary_by_seed.csv", all_summary)
    write_csv(out_dir / "online_summary_overall.csv", overall)
    write_csv(out_dir / "online_main_comparison.csv", main_rows)
    write_csv(out_dir / "online_mechanism_gains.csv", gains)
    write_csv(out_dir / "online_gate_table.csv", gate_rows)

    manifest = {
        "run_id": f"flecar_online_{len(seeds)}seeds_{len(files)}instances",
        "n_instances": len(files),
        "events_per_instance": int(args.events_per_instance),
        "seeds": seeds,
        "total_events": len(files) * int(args.events_per_instance) * len(seeds),
        "methods": sorted({str(r["method"]) for r in all_rows}),
        "outputs": {
            "event_rows": str(out_dir / "online_event_rows.csv"),
            "summary_overall": str(out_dir / "online_summary_overall.csv"),
            "main_comparison": str(out_dir / "online_main_comparison.csv"),
            "mechanism_gains": str(out_dir / "online_mechanism_gains.csv"),
            "gate_table": str(out_dir / "online_gate_table.csv"),
        },
    }
    write_json(out_dir / "manifest.json", manifest)
    write_json(ROOT / "research_flecar/outputs/latest_online_manifest.json", manifest)
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
