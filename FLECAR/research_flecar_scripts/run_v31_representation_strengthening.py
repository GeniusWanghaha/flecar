from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from statistics import mean
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC_Q = ROOT / "src"
SRC_F = ROOT / "research_flecar" / "src"
for p in [SRC_Q, SRC_F]:
    if str(p) not in sys.path:
        sys.path.append(str(p))

from flecar.evaluation import OnlineRunConfig, run_online_block  # noqa: E402
from flecar.utils import list_instance_files, read_json, write_csv, write_json  # noqa: E402


METHOD_A = "fes_car"
METHOD_B = "point_car_like_repair"


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
        }
    a = a_vals[:n]
    b = b_vals[:n]
    deltas = np.array([x - y for x, y in zip(a, b)], dtype=float)
    rng = np.random.default_rng(seed)
    boot_idx = rng.integers(0, n, size=(4000, n))
    boot_means = deltas[boot_idx].mean(axis=1)
    return {
        "n_pairs": int(n),
        "mean_a": float(mean(a)),
        "mean_b": float(mean(b)),
        "paired_delta": float(deltas.mean()),
        "delta_ci95_low": float(np.quantile(boot_means, 0.025)),
        "delta_ci95_high": float(np.quantile(boot_means, 0.975)),
    }


def _theme() -> dict[str, str]:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "axes.grid": True,
            "grid.color": "#D8DDD8",
            "grid.linestyle": "--",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.65,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 1.0,
        }
    )
    return {
        "ink": "#27313A",
        "slate": "#7E94A3",
        "teal": "#5E8C87",
        "sage": "#A8BFA0",
        "sand": "#D7C3A4",
        "orange": "#C98A6A",
        "red": "#B86F6B",
        "panel": "#F7F6F2",
        "grid": "#D8DDD8",
    }


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=360, bbox_inches="tight")
    if path.suffix.lower() == ".png":
        fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _metric_table(df: pd.DataFrame, slice_name: str, metrics: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metric in metrics:
        ps = _paired_stats(
            df[df["method"] == METHOD_A][metric].astype(float).tolist(),
            df[df["method"] == METHOD_B][metric].astype(float).tolist(),
        )
        rows.append(
            {
                "slice": slice_name,
                "metric": metric,
                "mean_fes_car": ps["mean_a"],
                "mean_point_car_like_repair": ps["mean_b"],
                "delta_fes_minus_point_car_like": ps["paired_delta"],
                "delta_ci95_low": ps["delta_ci95_low"],
                "delta_ci95_high": ps["delta_ci95_high"],
                "n_pairs": ps["n_pairs"],
            }
        )
    return rows


def _run_budget_probe(files: list[Path], budget: float, seeds: list[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        cfg = OnlineRunConfig(
            top_k=2,
            density_scale=1.5,
            density_seed=seed,
            time_budget_sec=float(budget),
            events_per_instance=4,
            seed=seed,
        )
        rr = run_online_block(files, cfg)
        for r in rr:
            rows.append({**r, "seed": seed, "time_budget_sec": float(budget)})
    return rows


def _canonical_and_slice_tables(v30_rows: pd.DataFrame, manifest: pd.DataFrame) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    flags = manifest.copy()
    flags["episode_id"] = flags["episode_id"].astype(str)
    merged = v30_rows.copy()
    merged["episode_id"] = merged["episode_id"].astype(str)
    merged = merged.merge(
        flags[
            [
                "episode_id",
                "is_higher_conflict_split",
                "is_larger_graph_split",
            ]
        ],
        on="episode_id",
        how="left",
    )
    merged["is_higher_conflict_split"] = merged["is_higher_conflict_split"].fillna(0).astype(int)
    merged["is_larger_graph_split"] = merged["is_larger_graph_split"].fillna(0).astype(int)

    keep = merged[merged["method"].isin([METHOD_A, METHOD_B])].copy()
    metrics = [
        "post_event_feasible",
        "future_feasibility_after_next_event",
        "repair_free_absorption",
        "repair_triggered",
        "repair_burden",
        "envelope_churn",
        "edit_burden",
        "repair_latency",
    ]
    rows: list[dict[str, Any]] = []
    rows += _metric_table(keep, "canonical_all_events", metrics)
    rows += _metric_table(keep[keep["event_type"] == "task_delay"], "task_delay", metrics)
    rows += _metric_table(
        keep[keep["is_higher_conflict_split"] == 1],
        "higher_conflict",
        metrics,
    )
    rows += _metric_table(
        keep[(keep["is_higher_conflict_split"] == 1) & (keep["is_larger_graph_split"] == 1)],
        "higher_conflict_and_larger_graph",
        metrics,
    )
    rows += _metric_table(
        keep[keep["event_type"] == "resource_occupation_change"],
        "resource_occupation_change",
        metrics,
    )
    return rows, keep


def _build_figure(slice_rows: pd.DataFrame, budget_rows: pd.DataFrame, out_path: Path) -> None:
    colors = _theme()
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6), constrained_layout=True)

    panel_a = slice_rows[
        (slice_rows["slice"] == "canonical_all_events")
        & (slice_rows["metric"].isin(["repair_free_absorption", "repair_triggered", "post_event_feasible"]))
    ].copy()
    order_a = ["repair_free_absorption", "repair_triggered", "post_event_feasible"]
    labels_a = {
        "repair_free_absorption": "Repair-free\nabsorption",
        "repair_triggered": "Repair\ntriggered ↓",
        "post_event_feasible": "Post-event\nfeasible",
    }
    panel_a["metric"] = pd.Categorical(panel_a["metric"], order_a)
    panel_a = panel_a.sort_values("metric")
    x = np.arange(len(panel_a))
    w = 0.36
    axes[0].bar(x - w / 2, panel_a["mean_fes_car"], width=w, color=colors["teal"], label="FES+CAR")
    axes[0].bar(x + w / 2, panel_a["mean_point_car_like_repair"], width=w, color=colors["slate"], label="Point+CAR-like")
    axes[0].set_xticks(x, [labels_a[m] for m in panel_a["metric"]])
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_title("(a) Representation-side absorbability")
    axes[0].set_ylabel("Rate")
    axes[0].legend(loc="upper left", frameon=False)

    panel_b = slice_rows[
        (slice_rows["slice"] == "canonical_all_events")
        & (slice_rows["metric"].isin(["repair_burden", "envelope_churn", "edit_burden", "repair_latency"]))
    ].copy()
    order_b = ["repair_burden", "envelope_churn", "edit_burden", "repair_latency"]
    labels_b = {
        "repair_burden": "Norm.\nburden",
        "envelope_churn": "Envelope\nchurn",
        "edit_burden": "Edit\nburden",
        "repair_latency": "Repair\nlatency",
    }
    panel_b["metric"] = pd.Categorical(panel_b["metric"], order_b)
    panel_b = panel_b.sort_values("metric")
    x = np.arange(len(panel_b))
    axes[1].bar(x - w / 2, panel_b["mean_fes_car"], width=w, color=colors["teal"])
    axes[1].bar(x + w / 2, panel_b["mean_point_car_like_repair"], width=w, color=colors["slate"])
    axes[1].set_xticks(x, [labels_b[m] for m in panel_b["metric"]])
    axes[1].set_title("(b) Locality and burden under matched repair")

    panel_c = budget_rows[budget_rows["metric"].isin(["post_event_feasible", "repair_burden"])].copy()
    budgets = sorted(panel_c["budget"].unique())
    for metric, color, ls, marker in [
        ("post_event_feasible", colors["orange"], "-", "o"),
        ("repair_burden", colors["red"], "--", "s"),
    ]:
        sub = panel_c[panel_c["metric"] == metric].sort_values("budget")
        axes[2].plot(
            sub["budget"],
            sub["delta_fes_minus_point_car_like"],
            color=color,
            linestyle=ls,
            marker=marker,
            linewidth=2.0,
            label="Post-event delta" if metric == "post_event_feasible" else "Burden delta",
        )
        axes[2].fill_between(
            sub["budget"],
            sub["delta_ci95_low"],
            sub["delta_ci95_high"],
            color=color,
            alpha=0.14,
        )
    axes[2].axhline(0.0, color=colors["ink"], linewidth=0.8)
    axes[2].set_xticks(budgets, [f"{b:.2f}s" for b in budgets])
    axes[2].set_title("(c) Tight-budget stress")
    axes[2].set_ylabel("Paired delta: FES+CAR - Point+CAR-like")
    axes[2].legend(loc="best", frameon=False)

    for ax in axes:
        ax.spines["left"].set_color(colors["ink"])
        ax.spines["bottom"].set_color(colors["ink"])

    _save(fig, out_path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances-dir", type=Path, default=Path("research_flecar/data_build/flecar_instances_v1"))
    ap.add_argument("--reference-online-rows", type=Path, default=Path("research_flecar/outputs_v30/raw/online/online_event_rows.csv"))
    ap.add_argument("--instance-manifest", type=Path, default=Path("research_flecar/data_build/flecar_instances_v1/instance_manifest.csv"))
    ap.add_argument("--out-root", type=Path, default=Path("research_flecar/outputs_v31_strengthening"))
    ap.add_argument("--seeds", type=str, default="7,11,19,23,29")
    ap.add_argument("--budget-values", type=str, default="0.05,0.50")
    ap.add_argument("--max-budget-instances", type=int, default=50)
    args = ap.parse_args()

    out_root = ROOT / args.out_root
    raw_dir = out_root / "raw"
    tab_dir = out_root / "exports" / "tables"
    fig_dir = out_root / "exports" / "figures"
    raw_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    v30_rows = pd.read_csv(ROOT / args.reference_online_rows)
    manifest = pd.read_csv(ROOT / args.instance_manifest)
    slice_rows, keep = _canonical_and_slice_tables(v30_rows, manifest)
    slice_df = pd.DataFrame(slice_rows)
    write_csv(tab_dir / "table_representation_slice_summary.csv", slice_rows)
    write_csv(raw_dir / "canonical_representation_rows.csv", keep.to_dict("records"))

    files = list_instance_files(ROOT / args.instances_dir)[: int(args.max_budget_instances)]
    seeds = [int(x.strip()) for x in str(args.seeds).split(",") if x.strip()]
    budgets = [float(x.strip()) for x in str(args.budget_values).split(",") if x.strip()]
    budget_raw_rows: list[dict[str, Any]] = []
    budget_summary_rows: list[dict[str, Any]] = []

    for budget in budgets:
        rr = _run_budget_probe(files, budget, seeds)
        budget_raw_rows.extend(rr)
        d = pd.DataFrame(rr)
        d = d[d["method"].isin([METHOD_A, METHOD_B])].copy()
        for metric in [
            "post_event_feasible",
            "future_feasibility_after_next_event",
            "repair_free_absorption",
            "repair_triggered",
            "repair_burden",
            "envelope_churn",
            "edit_burden",
            "repair_latency",
        ]:
            ps = _paired_stats(
                d[d["method"] == METHOD_A][metric].astype(float).tolist(),
                d[d["method"] == METHOD_B][metric].astype(float).tolist(),
            )
            budget_summary_rows.append(
                {
                    "budget": float(budget),
                    "metric": metric,
                    "mean_fes_car": ps["mean_a"],
                    "mean_point_car_like_repair": ps["mean_b"],
                    "delta_fes_minus_point_car_like": ps["paired_delta"],
                    "delta_ci95_low": ps["delta_ci95_low"],
                    "delta_ci95_high": ps["delta_ci95_high"],
                    "n_pairs": ps["n_pairs"],
                }
            )

    write_csv(raw_dir / "budget_stress_rows.csv", budget_raw_rows)
    write_csv(tab_dir / "table_representation_budget_stress.csv", budget_summary_rows)

    headline = []
    for lookup in [
        ("canonical_all_events", "repair_free_absorption"),
        ("canonical_all_events", "repair_burden"),
        ("higher_conflict", "future_feasibility_after_next_event"),
        ("task_delay", "repair_free_absorption"),
    ]:
        row = slice_df[(slice_df["slice"] == lookup[0]) & (slice_df["metric"] == lookup[1])].iloc[0]
        headline.append(
            {
                "block": "slice_summary",
                "slice": lookup[0],
                "metric": lookup[1],
                "mean_fes_car": float(row["mean_fes_car"]),
                "mean_point_car_like_repair": float(row["mean_point_car_like_repair"]),
                "delta_fes_minus_point_car_like": float(row["delta_fes_minus_point_car_like"]),
                "delta_ci95_low": float(row["delta_ci95_low"]),
                "delta_ci95_high": float(row["delta_ci95_high"]),
                "n_pairs": int(row["n_pairs"]),
            }
        )
    bdf = pd.DataFrame(budget_summary_rows)
    for budget in budgets:
        row = bdf[(bdf["budget"] == budget) & (bdf["metric"] == "repair_burden")].iloc[0]
        headline.append(
            {
                "block": "budget_stress",
                "slice": f"budget_{budget:.2f}",
                "metric": "repair_burden",
                "mean_fes_car": float(row["mean_fes_car"]),
                "mean_point_car_like_repair": float(row["mean_point_car_like_repair"]),
                "delta_fes_minus_point_car_like": float(row["delta_fes_minus_point_car_like"]),
                "delta_ci95_low": float(row["delta_ci95_low"]),
                "delta_ci95_high": float(row["delta_ci95_high"]),
                "n_pairs": int(row["n_pairs"]),
            }
        )
    write_csv(tab_dir / "table_representation_headline_claims.csv", headline)

    _build_figure(slice_df, bdf, fig_dir / "fig_v31_representation_value.png")

    manifest_out = {
        "run_id": "v31_representation_strengthening",
        "reference_online_rows": str((ROOT / args.reference_online_rows).resolve()),
        "instance_manifest": str((ROOT / args.instance_manifest).resolve()),
        "budget_values": budgets,
        "budget_instance_count": int(args.max_budget_instances),
        "seeds": seeds,
        "methods_compared": [METHOD_A, METHOD_B],
        "outputs": {
            "table_representation_slice_summary": str((tab_dir / "table_representation_slice_summary.csv").resolve()),
            "table_representation_budget_stress": str((tab_dir / "table_representation_budget_stress.csv").resolve()),
            "table_representation_headline_claims": str((tab_dir / "table_representation_headline_claims.csv").resolve()),
            "figure_representation_value": str((fig_dir / "fig_v31_representation_value.png").resolve()),
        },
    }
    write_json(out_root / "manifest.json", manifest_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
