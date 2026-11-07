from __future__ import annotations

import random
from pathlib import Path
from statistics import mean

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "research_flecar/outputs_stronger_evidence"
RAW = OUT / "raw"
TAB = OUT / "exports/tables"


def _bootstrap_mean_ci(deltas: list[float], seed: int = 7, n_boot: int = 2000) -> tuple[float, float]:
    if not deltas:
        return 0.0, 0.0
    rng = random.Random(seed)
    n = len(deltas)
    boots = []
    for _ in range(n_boot):
        sample = [deltas[rng.randrange(n)] for _ in range(n)]
        boots.append(sum(sample) / max(1, len(sample)))
    boots.sort()
    lo = boots[max(0, int(0.025 * len(boots)) - 1)]
    hi = boots[min(len(boots) - 1, int(0.975 * len(boots)) - 1)]
    return float(lo), float(hi)


def _paired_stats(a: list[float], b: list[float]) -> dict[str, float]:
    n = min(len(a), len(b))
    a = a[:n]
    b = b[:n]
    deltas = [x - y for x, y in zip(a, b)]
    lo, hi = _bootstrap_mean_ci(deltas, seed=7)
    return {
        "n_pairs": n,
        "mean_a": float(mean(a)) if a else 0.0,
        "mean_b": float(mean(b)) if b else 0.0,
        "paired_delta": float(mean(deltas)) if deltas else 0.0,
        "delta_ci95_low": float(lo),
        "delta_ci95_high": float(hi),
    }


def main() -> int:
    # table_longer_chain_results
    long_rows = []
    canonical_ch = pd.read_csv(ROOT / "research_flecar/outputs_post_cleanup/raw/chained/chained_event_rows.csv")
    for cl in [2, 3, 4]:
        d = canonical_ch[(canonical_ch["chain_len"] == cl) & (canonical_ch["event_idx"] == cl - 1)]
        ps = _paired_stats(
            d[d["method"] == "fes_car"]["post_event_feasible"].astype(float).tolist(),
            d[d["method"] == "point_legacy_repair"]["post_event_feasible"].astype(float).tolist(),
        )
        long_rows.append(
            {
                "chain_len": cl,
                "source": "post_cleanup",
                "fes_car_terminal_success": ps["mean_a"],
                "point_legacy_terminal_success": ps["mean_b"],
                "delta_fes_minus_point_legacy": ps["paired_delta"],
                "delta_ci95_low": ps["delta_ci95_low"],
                "delta_ci95_high": ps["delta_ci95_high"],
                "n_pairs": int(ps["n_pairs"]),
            }
        )
    for cl in [5, 6]:
        d = pd.read_csv(RAW / f"ep2_chain_{cl}_rows.csv")
        d = d[d["event_idx"] == cl - 1]
        ps = _paired_stats(
            d[d["method"] == "fes_car"]["post_event_feasible"].astype(float).tolist(),
            d[d["method"] == "point_legacy_repair"]["post_event_feasible"].astype(float).tolist(),
        )
        long_rows.append(
            {
                "chain_len": cl,
                "source": "stronger_evidence",
                "fes_car_terminal_success": ps["mean_a"],
                "point_legacy_terminal_success": ps["mean_b"],
                "delta_fes_minus_point_legacy": ps["paired_delta"],
                "delta_ci95_low": ps["delta_ci95_low"],
                "delta_ci95_high": ps["delta_ci95_high"],
                "n_pairs": int(ps["n_pairs"]),
            }
        )
    pd.DataFrame(long_rows).to_csv(TAB / "table_longer_chain_results.csv", index=False)

    # table_ood_harder_results
    ood_rows = []
    for name in ["harder_resource_sync_stress", "larger_graph_scaleup", "ood_skewed_disturbance_mix"]:
        d = pd.read_csv(RAW / f"ep2_{name}_rows.csv")
        for metric in ["post_event_feasible", "future_feasibility_after_next_event", "repair_regret", "envelope_churn", "edit_burden"]:
            ps = _paired_stats(
                d[d["method"] == "fes_car"][metric].astype(float).tolist(),
                d[d["method"] == "point_legacy_repair"][metric].astype(float).tolist(),
            )
            ood_rows.append(
                {
                    "scenario": name,
                    "metric": metric,
                    "mean_fes_car": ps["mean_a"],
                    "mean_point_legacy": ps["mean_b"],
                    "delta_fes_minus_point_legacy": ps["paired_delta"],
                    "delta_ci95_low": ps["delta_ci95_low"],
                    "delta_ci95_high": ps["delta_ci95_high"],
                    "n_pairs": int(ps["n_pairs"]),
                }
            )
    pd.DataFrame(ood_rows).to_csv(TAB / "table_ood_harder_results.csv", index=False)

    # table_budget_sensitivity
    budget_rows = []
    dseed = pd.read_csv(RAW / "ep3_seed_expansion_rows.csv")
    for metric in ["post_event_feasible", "future_feasibility_after_next_event", "repair_regret", "envelope_churn", "edit_burden"]:
        ps = _paired_stats(
            dseed[dseed["method"] == "fes_car"][metric].astype(float).tolist(),
            dseed[dseed["method"] == "point_legacy_repair"][metric].astype(float).tolist(),
        )
        budget_rows.append(
            {
                "sensitivity_axis": "seed_expansion_10",
                "setting": "seeds_10",
                "metric": metric,
                "mean_fes_car": ps["mean_a"],
                "mean_point_legacy": ps["mean_b"],
                "delta_fes_minus_point_legacy": ps["paired_delta"],
                "delta_ci95_low": ps["delta_ci95_low"],
                "delta_ci95_high": ps["delta_ci95_high"],
                "n_pairs": int(ps["n_pairs"]),
            }
        )
    for b in ["0p3", "0p5", "0p8"]:
        d = pd.read_csv(RAW / f"ep3_budget_{b}_rows.csv")
        bb = b.replace("p", ".")
        for metric in ["post_event_feasible", "future_feasibility_after_next_event", "repair_regret", "envelope_churn", "edit_burden"]:
            ps = _paired_stats(
                d[d["method"] == "fes_car"][metric].astype(float).tolist(),
                d[d["method"] == "point_legacy_repair"][metric].astype(float).tolist(),
            )
            budget_rows.append(
                {
                    "sensitivity_axis": "budget_sweep",
                    "setting": f"budget_{bb}",
                    "metric": metric,
                    "mean_fes_car": ps["mean_a"],
                    "mean_point_legacy": ps["mean_b"],
                    "delta_fes_minus_point_legacy": ps["paired_delta"],
                    "delta_ci95_low": ps["delta_ci95_low"],
                    "delta_ci95_high": ps["delta_ci95_high"],
                    "n_pairs": int(ps["n_pairs"]),
                }
            )
    pd.DataFrame(budget_rows).to_csv(TAB / "table_budget_sensitivity.csv", index=False)

    # table_weight_threshold_sensitivity (only topk_density and car_weights recomputed)
    old = pd.read_csv(TAB / "table_weight_threshold_sensitivity.csv")
    new_rows = []
    for topk, dens in [(1, "1p2"), (2, "1p5"), (2, "1p8"), (3, "1p8")]:
        d = pd.read_csv(RAW / f"ep3_topk{topk}_dens{dens}_rows.csv")
        ps = _paired_stats(
            d[d["method"] == "fes_car"]["post_event_feasible"].astype(float).tolist(),
            d[d["method"] == "point_legacy_repair"]["post_event_feasible"].astype(float).tolist(),
        )
        new_rows.append(
            {
                "sensitivity_axis": "topk_density",
                "setting": f"topk_{topk}_dens_{dens.replace('p', '.')}",
                "metric": "post_event_feasible",
                "mean_fes_car": ps["mean_a"],
                "mean_point_legacy": ps["mean_b"],
                "delta_fes_minus_point_legacy": ps["paired_delta"],
                "delta_ci95_low": ps["delta_ci95_low"],
                "delta_ci95_high": ps["delta_ci95_high"],
                "n_pairs": int(ps["n_pairs"]),
            }
        )
    for lc, lm in [("1p2", "1p0"), ("2p0", "1p5"), ("3p0", "2p0")]:
        d = pd.read_csv(RAW / f"ep3_weight_lc{lc}_lm{lm}_rows.csv")
        ps = _paired_stats(
            d[d["method"] == "fes_car"]["future_feasibility_after_next_event"].astype(float).tolist(),
            d[d["method"] == "point_legacy_repair"]["future_feasibility_after_next_event"].astype(float).tolist(),
        )
        new_rows.append(
            {
                "sensitivity_axis": "car_weights",
                "setting": f"lambda_c_{lc.replace('p', '.')}_lambda_m_{lm.replace('p', '.')}",
                "metric": "future_feasibility_after_next_event",
                "mean_fes_car": ps["mean_a"],
                "mean_point_legacy": ps["mean_b"],
                "delta_fes_minus_point_legacy": ps["paired_delta"],
                "delta_ci95_low": ps["delta_ci95_low"],
                "delta_ci95_high": ps["delta_ci95_high"],
                "n_pairs": int(ps["n_pairs"]),
            }
        )
    thresh = old[old["sensitivity_axis"] == "stnu_threshold"].copy()
    out = pd.concat([pd.DataFrame(new_rows), thresh], ignore_index=True)
    out.to_csv(TAB / "table_weight_threshold_sensitivity.csv", index=False)

    print("recomputed CI tables in outputs_stronger_evidence")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

