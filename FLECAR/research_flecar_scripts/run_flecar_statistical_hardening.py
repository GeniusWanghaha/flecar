from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC_F = ROOT / "research_flecar" / "src"
if str(SRC_F) not in sys.path:
    sys.path.append(str(SRC_F))

from flecar.analysis.stats import bootstrap_ci, exact_sign_test_p_two_sided  # noqa: E402
from flecar.utils import read_csv, write_csv, write_json  # noqa: E402


def _f(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except Exception:
        return float(default)


def _std(vals: list[float]) -> float:
    n = len(vals)
    if n <= 1:
        return 0.0
    mu = sum(vals) / n
    return float(math.sqrt(sum((v - mu) ** 2 for v in vals) / (n - 1)))


def _stderr(vals: list[float]) -> float:
    n = len(vals)
    if n <= 1:
        return 0.0
    return float(_std(vals) / math.sqrt(n))


def _pairs(
    rows: list[dict[str, Any]],
    *,
    method_a: str,
    method_b: str,
    metric: str,
    key_fields: list[str],
    filter_fn: Any | None = None,
) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, ...], dict[str, Any]] = {}
    for r in rows:
        if filter_fn is not None and not bool(filter_fn(r)):
            continue
        m = str(r.get("method", ""))
        if m not in {method_a, method_b}:
            continue
        if metric not in r:
            continue
        k = tuple(str(r.get(x, "")) for x in key_fields)
        slot = by_key.setdefault(k, {"key": k, "a": None, "b": None, "seed": str(r.get("seed", ""))})
        for idx, kf in enumerate(key_fields):
            slot[kf] = k[idx]
        if m == method_a:
            slot["a"] = _f(r, metric)
        elif m == method_b:
            slot["b"] = _f(r, metric)

    out = []
    for v in by_key.values():
        if v["a"] is None or v["b"] is None:
            continue
        out.append(v)
    return out


def _paired_stats(
    pair_entries: list[dict[str, Any]],
    *,
    table: str,
    comparison: str,
    dataset: str,
    metric: str,
    pairing_unit: str,
    method_a: str,
    method_b: str,
    n_instances: int,
    n_events: int,
) -> dict[str, Any]:
    a_vals = [float(x["a"]) for x in pair_entries]
    b_vals = [float(x["b"]) for x in pair_entries]
    deltas = [float(x["a"]) - float(x["b"]) for x in pair_entries]

    wins = sum(1 for d in deltas if d > 1e-12)
    losses = sum(1 for d in deltas if d < -1e-12)
    ties = len(deltas) - wins - losses
    lo, hi = bootstrap_ci(deltas, n_boot=4000, alpha=0.05, seed=7) if deltas else (0.0, 0.0)
    pval = exact_sign_test_p_two_sided(wins, losses)

    by_seed_delta: dict[str, list[float]] = defaultdict(list)
    by_seed_a: dict[str, list[float]] = defaultdict(list)
    by_seed_b: dict[str, list[float]] = defaultdict(list)
    for x in pair_entries:
        s = str(x.get("seed", ""))
        by_seed_delta[s].append(float(x["a"]) - float(x["b"]))
        by_seed_a[s].append(float(x["a"]))
        by_seed_b[s].append(float(x["b"]))
    seed_delta_means = [mean(v) for v in by_seed_delta.values()] if by_seed_delta else []
    seed_a_means = [mean(v) for v in by_seed_a.values()] if by_seed_a else []
    seed_b_means = [mean(v) for v in by_seed_b.values()] if by_seed_b else []

    return {
        "table": table,
        "comparison": comparison,
        "dataset": dataset,
        "metric": metric,
        "pairing_unit": pairing_unit,
        "method_a": method_a,
        "method_b": method_b,
        "n_pairs": len(pair_entries),
        "n_instances": int(n_instances),
        "n_events": int(n_events),
        "n_seeds": len(by_seed_delta),
        "mean_a": mean(a_vals) if a_vals else 0.0,
        "std_a": _std(a_vals),
        "stderr_a": _stderr(a_vals),
        "mean_b": mean(b_vals) if b_vals else 0.0,
        "std_b": _std(b_vals),
        "stderr_b": _stderr(b_vals),
        "paired_delta_mean": mean(deltas) if deltas else 0.0,
        "delta_ci95_low": float(lo),
        "delta_ci95_high": float(hi),
        "wins": int(wins),
        "losses": int(losses),
        "ties": int(ties),
        "n_non_ties": int(wins + losses),
        "p_two_sided_exact_sign_test": float(pval),
        "seed_delta_mean": mean(seed_delta_means) if seed_delta_means else 0.0,
        "seed_delta_std": _std(seed_delta_means),
        "seed_delta_stderr": _stderr(seed_delta_means),
        "seed_mean_a": mean(seed_a_means) if seed_a_means else 0.0,
        "seed_mean_b": mean(seed_b_means) if seed_b_means else 0.0,
        "seed_std_a": _std(seed_a_means),
        "seed_std_b": _std(seed_b_means),
    }


def _terminal_rows(rows: list[dict[str, Any]], extra_keys: list[str] | None = None) -> list[dict[str, Any]]:
    ek = extra_keys or []
    best: dict[tuple[str, ...], dict[str, Any]] = {}
    for r in rows:
        base_key = [str(r.get("seed", "")), str(r.get("episode_id", "")), str(r.get("method", ""))]
        base_key.extend(str(r.get(k, "")) for k in ek)
        k = tuple(base_key)
        cur = best.get(k)
        if cur is None or int(_f(r, "event_idx", 0.0)) > int(_f(cur, "event_idx", 0.0)):
            best[k] = r
    return list(best.values())


def _episode_split_flags(manifest_rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for r in manifest_rows:
        eid = str(r.get("episode_id", ""))
        if not eid:
            continue
        out[eid] = {
            "is_larger_graph_split": int(float(r.get("is_larger_graph_split", 0) or 0)),
            "is_higher_candidate_density_split": int(float(r.get("is_higher_candidate_density_split", 0) or 0)),
            "is_higher_conflict_split": int(float(r.get("is_higher_conflict_split", 0) or 0)),
            "is_chained_disturbance_split": int(float(r.get("is_chained_disturbance_split", 0) or 0)),
        }
    return out


def _attach_split_flags(rows: list[dict[str, Any]], split_map: dict[str, dict[str, int]]) -> list[dict[str, Any]]:
    out = []
    for r in rows:
        eid = str(r.get("episode_id", ""))
        flags = split_map.get(eid, {})
        out.append(
            {
                **r,
                "is_larger_graph_split": int(flags.get("is_larger_graph_split", 0)),
                "is_higher_candidate_density_split": int(flags.get("is_higher_candidate_density_split", 0)),
                "is_higher_conflict_split": int(flags.get("is_higher_conflict_split", 0)),
                "is_chained_disturbance_split": int(flags.get("is_chained_disturbance_split", 0)),
            }
        )
    return out


def _pick_strongest_point_baseline(static_rows: list[dict[str, Any]]) -> str:
    # Restrict to static-main point baselines used in the main comparison block.
    point_methods = ["explicit_scheduling", "explicit_scheduling_bacb"]
    means: list[tuple[float, str]] = []
    for m in point_methods:
        vals = [_f(r, "feasible") for r in static_rows if str(r.get("method", "")) == m]
        if not vals:
            continue
        means.append((mean(vals), m))
    if not means:
        return "explicit_scheduling_bacb"
    means.sort(reverse=True)
    return means[0][1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs-root", type=Path, default=Path("research_flecar/outputs"))
    ap.add_argument("--manifest-csv", type=Path, default=Path("research_flecar/data_build/flecar_instances_v1/instance_manifest.csv"))
    ap.add_argument("--out-dir", type=Path, default=Path("research_flecar/outputs/exports/tables"))
    args = ap.parse_args()

    outputs_root = ROOT / args.outputs_root
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    static_rows = read_csv(outputs_root / "static" / "static_rows.csv")
    online_rows = read_csv(outputs_root / "online" / "online_event_rows.csv")
    chained_rows = read_csv(outputs_root / "chained" / "chained_event_rows.csv")
    manifest_rows = read_csv(ROOT / args.manifest_csv)
    split_map = _episode_split_flags(manifest_rows)

    static_rows = _attach_split_flags(static_rows, split_map)
    online_rows = _attach_split_flags(online_rows, split_map)
    chained_rows = _attach_split_flags(chained_rows, split_map)

    strongest_point = _pick_strongest_point_baseline(static_rows)

    # Table 1: static main stats.
    static_metrics = [
        "feasible",
        "dispatchability_success",
        "envelope_flexibility_volume",
        "critical_slack_margin",
        "absorbable_disturbance_radius",
    ]
    static_table: list[dict[str, Any]] = []
    for metric in static_metrics:
        pairs = _pairs(
            static_rows,
            method_a="fes",
            method_b=strongest_point,
            metric=metric,
            key_fields=["seed", "episode_id"],
        )
        static_table.append(
            _paired_stats(
                pairs,
                table="table_static_main_stats",
                comparison=f"fes_vs_{strongest_point}",
                dataset="static_main",
                metric=metric,
                pairing_unit="seed+episode_id",
                method_a="fes",
                method_b=strongest_point,
                n_instances=len({str(x.get("episode_id", "")) for x in pairs}),
                n_events=0,
            )
        )

    # Table 2: online main stats.
    online_event_metrics = [
        "post_event_feasible",
        "cumulative_completion",
        "repair_free_absorption",
        "controllability_preserved",
        "future_feasibility_after_next_event",
        "repair_regret",
        "repair_burden",
        "envelope_churn",
        "edit_burden",
        "repair_latency",
    ]
    online_terminal = _terminal_rows(online_rows)

    online_comparisons = [
        ("fes_car", "point_legacy_repair", "fes_car_vs_point_legacy_repair"),
        ("fes_car", "point_car_like_repair", "fes_car_vs_point_car_like_repair"),
    ]
    online_table: list[dict[str, Any]] = []
    for ma, mb, cname in online_comparisons:
        for metric in online_event_metrics:
            pairs = _pairs(
                online_rows,
                method_a=ma,
                method_b=mb,
                metric=metric,
                key_fields=["seed", "episode_id", "event_idx"],
            )
            if not pairs:
                continue
            online_table.append(
                _paired_stats(
                    pairs,
                    table="table_online_main_stats",
                    comparison=cname,
                    dataset="online_main_event",
                    metric=metric,
                    pairing_unit="seed+episode_id+event_idx",
                    method_a=ma,
                    method_b=mb,
                    n_instances=len({str(x.get("episode_id", "")) for x in pairs}),
                    n_events=len(pairs),
                )
            )

        for metric in ["terminal_success_proxy", "terminal_completion_proxy"]:
            pairs = _pairs(
                online_terminal,
                method_a=ma,
                method_b=mb,
                metric=metric,
                key_fields=["seed", "episode_id"],
            )
            if not pairs:
                continue
            online_table.append(
                _paired_stats(
                    pairs,
                    table="table_online_main_stats",
                    comparison=cname,
                    dataset="online_main_terminal",
                    metric=metric,
                    pairing_unit="seed+episode_id(last_event)",
                    method_a=ma,
                    method_b=mb,
                    n_instances=len({str(x.get("episode_id", "")) for x in pairs}),
                    n_events=0,
                )
            )

    # Table 3: harder split breakdown.
    split_defs = [
        ("is_larger_graph_split", "larger_graph"),
        ("is_higher_conflict_split", "higher_conflict"),
        ("is_higher_candidate_density_split", "higher_candidate_density"),
        ("is_chained_disturbance_split", "chained_disturbance"),
    ]
    harder_table: list[dict[str, Any]] = []
    for flag, label in split_defs:
        # static harder breakdown
        for metric in ["feasible", "dispatchability_success", "absorbable_disturbance_radius"]:
            pairs = _pairs(
                static_rows,
                method_a="fes",
                method_b=strongest_point,
                metric=metric,
                key_fields=["seed", "episode_id"],
                filter_fn=lambda r, ff=flag: int(_f(r, ff, 0.0)) == 1,
            )
            if not pairs:
                continue
            harder_table.append(
                {
                    "slice": label,
                    **_paired_stats(
                        pairs,
                        table="table_harder_split_stats",
                        comparison=f"fes_vs_{strongest_point}",
                        dataset="static_harder",
                        metric=metric,
                        pairing_unit="seed+episode_id",
                        method_a="fes",
                        method_b=strongest_point,
                        n_instances=len({str(x.get("episode_id", "")) for x in pairs}),
                        n_events=0,
                    ),
                }
            )

        # online harder breakdown
        for ma, mb, cname in online_comparisons:
            for metric in ["post_event_feasible", "future_feasibility_after_next_event", "repair_regret", "repair_burden"]:
                pairs = _pairs(
                    online_rows,
                    method_a=ma,
                    method_b=mb,
                    metric=metric,
                    key_fields=["seed", "episode_id", "event_idx"],
                    filter_fn=lambda r, ff=flag: int(_f(r, ff, 0.0)) == 1,
                )
                if not pairs:
                    continue
                harder_table.append(
                    {
                        "slice": label,
                        **_paired_stats(
                            pairs,
                            table="table_harder_split_stats",
                            comparison=cname,
                            dataset="online_harder",
                            metric=metric,
                            pairing_unit="seed+episode_id+event_idx",
                            method_a=ma,
                            method_b=mb,
                            n_instances=len({str(x.get("episode_id", "")) for x in pairs}),
                            n_events=len(pairs),
                        ),
                    }
                )

    # Table 4: chained stats.
    chained_table: list[dict[str, Any]] = []
    chained_terminal = _terminal_rows(chained_rows, extra_keys=["chain_len"])
    chain_lens = sorted({int(_f(r, "chain_len", 0.0)) for r in chained_rows})
    for cl in chain_lens:
        event_subset = [r for r in chained_rows if int(_f(r, "chain_len", 0.0)) == cl]
        term_subset = [r for r in chained_terminal if int(_f(r, "chain_len", 0.0)) == cl]
        for ma, mb, cname in online_comparisons:
            for metric in [
                "post_event_feasible",
                "future_feasibility_after_next_event",
                "repair_regret",
                "repair_burden",
                "envelope_churn",
                "edit_burden",
                "repair_latency",
            ]:
                pairs = _pairs(
                    event_subset,
                    method_a=ma,
                    method_b=mb,
                    metric=metric,
                    key_fields=["seed", "episode_id", "event_idx", "chain_len"],
                )
                if not pairs:
                    continue
                chained_table.append(
                    {
                        "chain_len": cl,
                        **_paired_stats(
                            pairs,
                            table="table_chained_stats",
                            comparison=cname,
                            dataset="chained_event",
                            metric=metric,
                            pairing_unit="seed+episode_id+event_idx+chain_len",
                            method_a=ma,
                            method_b=mb,
                            n_instances=len({str(x.get("episode_id", "")) for x in pairs}),
                            n_events=len(pairs),
                        ),
                    }
                )
            for metric in ["terminal_success_proxy", "terminal_completion_proxy"]:
                pairs = _pairs(
                    term_subset,
                    method_a=ma,
                    method_b=mb,
                    metric=metric,
                    key_fields=["seed", "episode_id", "chain_len"],
                )
                if not pairs:
                    continue
                chained_table.append(
                    {
                        "chain_len": cl,
                        **_paired_stats(
                            pairs,
                            table="table_chained_stats",
                            comparison=cname,
                            dataset="chained_terminal",
                            metric=metric,
                            pairing_unit="seed+episode_id(last_event)+chain_len",
                            method_a=ma,
                            method_b=mb,
                            n_instances=len({str(x.get("episode_id", "")) for x in pairs}),
                            n_events=0,
                        ),
                    }
                )

    write_csv(out_dir / "table_static_main_stats.csv", static_table)
    write_csv(out_dir / "table_online_main_stats.csv", online_table)
    write_csv(out_dir / "table_harder_split_stats.csv", harder_table)
    write_csv(out_dir / "table_chained_stats.csv", chained_table)

    manifest = {
        "run_id": "flecar_statistical_hardening_v1",
        "inputs": {
            "static_rows": str(outputs_root / "static" / "static_rows.csv"),
            "online_rows": str(outputs_root / "online" / "online_event_rows.csv"),
            "chained_rows": str(outputs_root / "chained" / "chained_event_rows.csv"),
            "instance_manifest": str(ROOT / args.manifest_csv),
        },
        "strongest_point_baseline_static": strongest_point,
        "outputs": {
            "table_static_main_stats": str(out_dir / "table_static_main_stats.csv"),
            "table_online_main_stats": str(out_dir / "table_online_main_stats.csv"),
            "table_harder_split_stats": str(out_dir / "table_harder_split_stats.csv"),
            "table_chained_stats": str(out_dir / "table_chained_stats.csv"),
        },
    }
    write_json(out_dir / "statistical_hardening_manifest.json", manifest)
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
