from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC_F = ROOT / "research_flecar" / "src"
if str(SRC_F) not in sys.path:
    sys.path.append(str(SRC_F))

from flecar.data import BuildConfig, build_flecar_dataset  # noqa: E402
from flecar.utils import list_instance_files, read_json, write_csv, write_json  # noqa: E402


def _quantile(vals: list[float], q: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    idx = int(round((len(s) - 1) * max(0.0, min(1.0, q))))
    return float(s[idx])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-instances", type=Path, default=Path("data/instances/real_partnr/oracle_static"))
    ap.add_argument("--out-dir", type=Path, default=Path("research_flecar/data_build/flecar_instances_v1"))
    ap.add_argument("--target-instances", type=int, default=360)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    cfg = BuildConfig(
        source_instances=ROOT / args.source_instances,
        out_dir=ROOT / args.out_dir,
        target_instances=int(args.target_instances),
        seed=int(args.seed),
    )
    summary = build_flecar_dataset(cfg)

    files = sorted(cfg.out_dir.glob('instance_*.json'))
    rows = []
    for idx, p in enumerate(files):
        inst = read_json(p)
        n_subtasks = len(inst.get("subtasks", []))
        n_cands = len(inst.get("candidate_assignments", []))
        density = float(n_cands) / float(max(1, n_subtasks))
        conflict_proxy = float(
            len(inst.get("precedence", []))
            + len(inst.get("sync_constraints", []))
            + len(inst.get("resource_conflicts", []))
            + len(inst.get("spatial_conflicts", []))
        )

        xi = inst.get("Xi", inst.get("metadata", {}).get("Xi", {}))
        chain_risk = float(
            xi.get("temporary_agent_unavailability", {}).get("event_probability", 0.0)
            + xi.get("resource_occupancy_fluctuation", {}).get("spike_probability", 0.0)
            + xi.get("task_failure_hazard_prior", {}).get("base_hazard", 0.0)
        )

        rows.append(
            {
                "instance_path": str(p),
                "episode_id": str(inst["episode_id"]),
                "n_subtasks": n_subtasks,
                "n_candidates": n_cands,
                "candidate_density": density,
                "conflict_proxy": conflict_proxy,
                "chained_risk_proxy": chain_risk,
                "split": "train" if idx < int(0.6 * len(files)) else ("val" if idx < int(0.8 * len(files)) else "test"),
                "is_larger_graph_split": int(n_subtasks >= 4),
            }
        )

    densities = [float(r["candidate_density"]) for r in rows]
    conflicts = [float(r["conflict_proxy"]) for r in rows]
    chain_scores = [float(r["chained_risk_proxy"]) for r in rows]
    thr_density = _quantile(densities, 0.80)
    thr_conflict = _quantile(conflicts, 0.80)
    thr_chain = _quantile(chain_scores, 0.75)

    for r in rows:
        r["is_higher_candidate_density_split"] = int(float(r["candidate_density"]) >= thr_density)
        r["is_higher_conflict_split"] = int(float(r["conflict_proxy"]) >= thr_conflict)
        r["is_chained_disturbance_split"] = int(float(r["chained_risk_proxy"]) >= thr_chain)

    write_csv(cfg.out_dir / "instance_manifest.csv", rows)

    out_manifest = {
        "run_id": "flecar_build_instances_v1",
        "source_instances": str(cfg.source_instances),
        "out_dir": str(cfg.out_dir),
        "target_instances": int(cfg.target_instances),
        "seed": int(cfg.seed),
        "summary": summary,
        "instance_manifest_csv": str(cfg.out_dir / "instance_manifest.csv"),
        "harder_split_thresholds": {
            "candidate_density_q80": thr_density,
            "conflict_proxy_q80": thr_conflict,
            "chained_risk_q75": thr_chain,
        },
    }
    write_json(ROOT / "research_flecar/outputs/latest_build_manifest.json", out_manifest)
    print(json.dumps(out_manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
