from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC_F = ROOT / "research_flecar" / "src"
if str(SRC_F) not in sys.path:
    sys.path.append(str(SRC_F))

from flecar.utils import read_csv, read_json, write_csv, write_json  # noqa: E402


def _float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except Exception:
        return float(default)


def _index(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    out = {}
    for r in rows:
        out[str(r.get(key, ""))] = r
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs-root", type=Path, default=Path("research_flecar/outputs"))
    ap.add_argument("--out-dir", type=Path, default=Path("research_flecar/outputs/exports"))
    args = ap.parse_args()

    outputs_root = ROOT / args.outputs_root
    out_dir = ROOT / args.out_dir
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    static_main = read_csv(outputs_root / "static" / "static_main_comparison.csv")
    static_harder = read_csv(outputs_root / "static" / "static_harder_slice_summary.csv")
    static_gate = read_csv(outputs_root / "static" / "static_threshold_table.csv")

    online_main = read_csv(outputs_root / "online" / "online_main_comparison.csv")
    online_gains = read_csv(outputs_root / "online" / "online_mechanism_gains.csv")
    online_gate = read_csv(outputs_root / "online" / "online_gate_table.csv")

    chained_overall = read_csv(outputs_root / "chained" / "chained_summary_overall.csv")
    chained_gains = read_csv(outputs_root / "chained" / "chained_gains.csv")

    adversarial_overall = read_csv(outputs_root / "adversarial" / "adversarial_summary_overall.csv")
    adversarial_gains = read_csv(outputs_root / "adversarial" / "adversarial_gains.csv")

    ablation_static = read_csv(outputs_root / "ablation" / "flecar_static_ablation_summary.csv")
    ablation_car = read_csv(outputs_root / "ablation" / "flecar_car_ablation_summary.csv")

    solver_summary = read_csv(outputs_root / "solver_generalization" / "solver_generalization_summary.csv")

    write_csv(tables_dir / "table_static_main.csv", static_main)
    write_csv(tables_dir / "table_static_harder.csv", static_harder)
    write_csv(tables_dir / "table_static_gate.csv", static_gate)

    write_csv(tables_dir / "table_online_main.csv", online_main)
    write_csv(tables_dir / "table_online_gains.csv", online_gains)
    write_csv(tables_dir / "table_online_gate.csv", online_gate)

    write_csv(tables_dir / "table_chained_main.csv", chained_overall)
    write_csv(tables_dir / "table_chained_gains.csv", chained_gains)

    write_csv(tables_dir / "table_adversarial_main.csv", adversarial_overall)
    write_csv(tables_dir / "table_adversarial_gains.csv", adversarial_gains)

    write_csv(tables_dir / "table_ablation_static.csv", ablation_static)
    write_csv(tables_dir / "table_ablation_car.csv", ablation_car)
    write_csv(tables_dir / "table_solver_generalization.csv", solver_summary)

    # Unified acceptance summary table.
    static_pass = int(static_gate[0].get("pass_overall_feasible_delta_ge_0_10", 0)) if static_gate else 0
    static_pass &= int(static_gate[0].get("pass_harder_delta_larger_than_overall", 0)) if static_gate else 0
    static_pass &= int(static_gate[0].get("pass_dispatchability_gain_ge_0_08", 0)) if static_gate else 0
    static_pass &= int(static_gate[0].get("pass_flex_or_csm_gain_ge_0_15", 0)) if static_gate else 0
    static_pass &= int(static_gate[0].get("pass_adr_gain_ge_0_20", 0)) if static_gate else 0

    online_gate_pass = int(online_gate[0].get("pass_online_gate_2_of_3", 0)) if online_gate else 0
    chained_terminal_pass = 0
    if chained_gains:
        chained_terminal_pass = int(any(int(float(r.get("pass_terminal_success_gain_ge_0_10", 0))) == 1 for r in chained_gains))

    build_manifest = read_json(outputs_root / "latest_build_manifest.json") if (outputs_root / "latest_build_manifest.json").exists() else {}
    static_manifest = read_json(outputs_root / "latest_static_manifest.json") if (outputs_root / "latest_static_manifest.json").exists() else {}
    online_manifest = read_json(outputs_root / "latest_online_manifest.json") if (outputs_root / "latest_online_manifest.json").exists() else {}

    scale_static = int(static_manifest.get("n_instances", 0))
    scale_online_events = int(online_manifest.get("total_events", 0))
    seeds = online_manifest.get("seeds", [])
    scale_pass = int(scale_static >= 300 and scale_online_events >= 1000 and len(seeds) >= 5)

    acceptance = [
        {
            "StaticGatePass": int(static_pass),
            "OnlineGatePass_2of3": int(online_gate_pass),
            "ChainedTerminalPass": int(chained_terminal_pass),
            "ScalePass": int(scale_pass),
            "StaticInstances": scale_static,
            "OnlineEvents": scale_online_events,
            "NumSeeds": len(seeds),
            "BuildRunId": str(build_manifest.get("run_id", "")),
            "StaticRunId": str(static_manifest.get("run_id", "")),
            "OnlineRunId": str(online_manifest.get("run_id", "")),
        }
    ]
    write_csv(tables_dir / "table_acceptance_gate.csv", acceptance)

    manifest = {
        "run_id": "flecar_export_tables_v1",
        "input_root": str(outputs_root),
        "table_dir": str(tables_dir),
        "tables_written": [
            "table_static_main.csv",
            "table_static_harder.csv",
            "table_static_gate.csv",
            "table_online_main.csv",
            "table_online_gains.csv",
            "table_online_gate.csv",
            "table_chained_main.csv",
            "table_chained_gains.csv",
            "table_adversarial_main.csv",
            "table_adversarial_gains.csv",
            "table_ablation_static.csv",
            "table_ablation_car.csv",
            "table_solver_generalization.csv",
            "table_acceptance_gate.csv",
        ],
    }
    write_json(out_dir / "tables_manifest.json", manifest)
    write_json(ROOT / "research_flecar/outputs/latest_tables_manifest.json", manifest)
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
