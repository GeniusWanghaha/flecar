from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from qcma.solvers.common import objective_from_selection


@dataclass
class EnvelopeMetrics:
    flexibility_volume: float
    critical_slack_margin: float
    sync_tolerance: float
    resource_free_buffer: float
    dispatchability_success: bool
    controllability_margin: float


def _pred_succ_maps(precedence: list[list[str]]) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    pred: dict[str, set[str]] = defaultdict(set)
    succ: dict[str, set[str]] = defaultdict(set)
    for pair in precedence:
        if not isinstance(pair, list) or len(pair) != 2:
            continue
        u = str(pair[0])
        v = str(pair[1])
        succ[u].add(v)
        pred[v].add(u)
        pred.setdefault(u, set())
        succ.setdefault(v, set())
    return pred, succ


def _selected_subtasks(reduced: dict[str, Any], selected_ids: list[str]) -> dict[str, dict[str, Any]]:
    nodes = reduced.get("nodes", {})
    selected = [str(cid) for cid in selected_ids if str(cid) in nodes]
    by_subtask: dict[str, dict[str, Any]] = {}
    for cid in selected:
        node = nodes[cid]
        by_subtask[str(node["subtask_id"])] = node
    return by_subtask


def _derive_sync_pairs(reduced: dict[str, Any]) -> list[list[str]]:
    if isinstance(reduced.get("sync_pairs"), list):
        out = []
        for pair in reduced["sync_pairs"]:
            if isinstance(pair, list) and len(pair) == 2:
                out.append([str(pair[0]), str(pair[1])])
        if out:
            return out
    nodes = reduced.get("nodes", {})
    derived: set[tuple[str, str]] = set()
    for edge in reduced.get("sync_invalid_edges", []):
        if not isinstance(edge, (list, tuple)) or len(edge) != 2:
            continue
        u_sid = str(nodes.get(str(edge[0]), {}).get("subtask_id", ""))
        v_sid = str(nodes.get(str(edge[1]), {}).get("subtask_id", ""))
        if not u_sid or not v_sid or u_sid == v_sid:
            continue
        key = (u_sid, v_sid) if u_sid < v_sid else (v_sid, u_sid)
        derived.add(key)
    return [[u, v] for (u, v) in sorted(derived)]


def _build_windows(
    by_subtask: dict[str, dict[str, Any]],
    precedence: list[list[str]],
    *,
    slack_expand_ratio: float,
    min_window_pad: int,
) -> dict[str, dict[str, float]]:
    pred, _ = _pred_succ_maps(precedence)
    windows: dict[str, dict[str, float]] = {}
    for sid, node in by_subtask.items():
        start = int(node["start"])
        dur = int(node["duration"])
        span = max(min_window_pad, int(round(float(dur) * float(slack_expand_ratio))))
        windows[sid] = {
            "l": float(max(0, start - span)),
            "u": float(start + span),
            "dur": float(dur),
            "nominal_start": float(start),
        }

    for _ in range(max(1, len(windows))):
        changed = False
        for sid, w in windows.items():
            old_l = w["l"]
            for psid in pred.get(sid, set()):
                if psid not in windows:
                    continue
                p = windows[psid]
                w["l"] = max(w["l"], p["l"] + p["dur"] - min_window_pad)
            if w["l"] > old_l + 1e-9:
                changed = True
        if not changed:
            break
    return windows


def _resource_non_overlap_pairs(by_subtask: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    by_agent: dict[str, list[tuple[int, int, str]]] = defaultdict(list)
    for sid, node in by_subtask.items():
        start = int(node["start"])
        end = start + int(node["duration"])
        by_agent[str(node["agent_id"])].append((start, end, sid))

    edges: list[dict[str, Any]] = []
    for aid, rows in by_agent.items():
        ordered = sorted(rows)
        for idx in range(1, len(ordered)):
            p = ordered[idx - 1]
            q = ordered[idx]
            edges.append(
                {
                    "u": str(p[2]),
                    "v": str(q[2]),
                    "type": "resource_non_overlap",
                    "agent_id": aid,
                }
            )
    return edges


def _dispatchability_components(
    windows: dict[str, dict[str, float]],
    precedence: list[list[str]],
    sync_pairs: list[list[str]],
    resource_pairs: list[dict[str, Any]],
) -> dict[str, Any]:
    prec_slacks_robust: list[float] = []
    prec_slacks_nominal: list[float] = []
    for pair in precedence:
        if not isinstance(pair, list) or len(pair) != 2:
            continue
        u, v = str(pair[0]), str(pair[1])
        if u not in windows or v not in windows:
            continue
        wu = windows[u]
        wv = windows[v]
        prec_slacks_robust.append(float(wv["l"] - (wu["u"] + wu["dur"])))
        prec_slacks_nominal.append(float(wv["u"] - (wu["l"] + wu["dur"])))

    sync_tols: list[float] = []
    for pair in sync_pairs:
        if not isinstance(pair, list) or len(pair) != 2:
            continue
        u, v = str(pair[0]), str(pair[1])
        if u not in windows or v not in windows:
            continue
        wu = windows[u]
        wv = windows[v]
        overlap = min(wu["u"], wv["u"]) - max(wu["l"], wv["l"])
        sync_tols.append(float(max(0.0, overlap)))

    resource_bufs: list[float] = []
    for edge in resource_pairs:
        u = str(edge.get("u", ""))
        v = str(edge.get("v", ""))
        if u not in windows or v not in windows:
            continue
        wu = windows[u]
        wv = windows[v]
        resource_bufs.append(float(wv["l"] - (wu["u"] + wu["dur"])))

    width_min = min((float(w["u"] - w["l"]) for w in windows.values()), default=0.0)
    robust_precedence_margin = min(prec_slacks_robust, default=0.0)
    critical_slack_margin = min(prec_slacks_nominal, default=0.0)
    sync_tolerance = min(sync_tols, default=0.0) if sync_tols else 0.0
    resource_free_buffer = min(resource_bufs, default=0.0) if resource_bufs else 0.0
    dispatchable = (
        all(float(w["u"]) >= float(w["l"]) for w in windows.values())
        and robust_precedence_margin >= -1e-6
        and sync_tolerance >= -1e-6
        and resource_free_buffer >= -1e-6
    )
    ctrl_margin = min(robust_precedence_margin, sync_tolerance, resource_free_buffer, width_min)
    return {
        "critical_slack_margin": float(critical_slack_margin),
        "robust_precedence_margin": float(robust_precedence_margin),
        "sync_tolerance": float(sync_tolerance),
        "resource_free_buffer": float(resource_free_buffer),
        "dispatchability_success": bool(dispatchable),
        "controllability_margin": float(ctrl_margin),
        "precedence_slacks_nominal": prec_slacks_nominal,
        "precedence_slacks_robust": prec_slacks_robust,
        "resource_slacks": resource_bufs,
        "window_width_min": float(width_min),
    }


def _envelope_validity(
    assignment: dict[str, str],
    windows: dict[str, dict[str, float]],
    precedence: list[list[str]],
) -> dict[str, Any]:
    reasons: list[str] = []
    if not assignment:
        reasons.append("empty_assignment")
    for sid, w in windows.items():
        if float(w["u"]) < float(w["l"]):
            reasons.append(f"invalid_window:{sid}")
    for pair in precedence:
        if not isinstance(pair, list) or len(pair) != 2:
            continue
        u, v = str(pair[0]), str(pair[1])
        if u in assignment and v in assignment and (u not in windows or v not in windows):
            reasons.append(f"missing_window_for_precedence:{u}->{v}")
    return {"valid": len(reasons) == 0, "reasons": reasons}


def envelope_objective(
    envelope: dict[str, Any],
    *,
    l_obj: float,
    lambda_obj: float,
    lambda_flex: float,
    lambda_risk: float,
) -> float:
    metrics = envelope.get("metrics", {})
    flex = float(metrics.get("flexibility_volume", 0.0))
    csm = float(metrics.get("critical_slack_margin", 0.0))
    sync_tol = float(metrics.get("sync_tolerance", 0.0))
    res_buf = float(metrics.get("resource_free_buffer", 0.0))
    ctrl = float(metrics.get("controllability_margin", 0.0))
    width_min = float(metrics.get("window_width_min", 0.0))

    phi = float(flex + 0.5 * max(0.0, csm) + 0.35 * sync_tol + 0.35 * res_buf + 0.2 * max(0.0, width_min))
    psi = float(
        max(0.0, -csm)
        + max(0.0, -sync_tol)
        + max(0.0, -res_buf)
        + max(0.0, -ctrl)
        + (0.2 if not bool(metrics.get("dispatchability_success", False)) else 0.0)
    )
    return float(lambda_obj * float(l_obj) - lambda_flex * phi + lambda_risk * psi)


def synthesize_envelope(
    reduced: dict[str, Any],
    selected_ids: list[str],
    *,
    slack_expand_ratio: float = 0.3,
    min_window_pad: int = 1,
    lambda_obj: float = 1.0,
    lambda_flex: float = 0.4,
    lambda_risk: float = 0.35,
    optimize_grid: bool = True,
) -> dict[str, Any]:
    by_subtask = _selected_subtasks(reduced, selected_ids)
    precedence = reduced.get("precedence_pairs", [])
    sync_pairs = _derive_sync_pairs(reduced)
    resource_pairs = _resource_non_overlap_pairs(by_subtask)
    selected_valid = [str(cid) for cid in selected_ids if str(cid) in reduced.get("nodes", {})]
    l_obj = objective_from_selection(reduced, selected_valid) if selected_valid else float("inf")

    ratio_grid = [float(slack_expand_ratio)]
    pad_grid = [int(min_window_pad)]
    if optimize_grid:
        ratio_grid = sorted(
            {
                max(0.05, float(slack_expand_ratio) * 0.7),
                max(0.05, float(slack_expand_ratio) * 0.9),
                max(0.05, float(slack_expand_ratio)),
                min(1.0, float(slack_expand_ratio) * 1.2),
                min(1.0, float(slack_expand_ratio) * 1.5),
            }
        )
        pad_grid = sorted({max(1, int(min_window_pad)), max(1, int(min_window_pad) + 1)})

    candidates: list[dict[str, Any]] = []
    for ratio in ratio_grid:
        for pad in pad_grid:
            windows = _build_windows(
                by_subtask,
                precedence,
                slack_expand_ratio=float(ratio),
                min_window_pad=int(pad),
            )
            dispatch_comp = _dispatchability_components(windows, precedence, sync_pairs, resource_pairs)
            flexibility_volume = sum(max(0.0, float(w["u"]) - float(w["l"])) for w in windows.values())
            horizon = 0.0
            for w in windows.values():
                horizon = max(horizon, float(w["u"] + w["dur"]))

            assignment = {sid: str(node["candidate_id"]) for sid, node in by_subtask.items()}
            validity = _envelope_validity(assignment, windows, precedence)

            envelope = {
                "x": assignment,
                "W": windows,
                "R": {
                    "precedence_pairs": [[str(p[0]), str(p[1])] for p in precedence if isinstance(p, list) and len(p) == 2],
                    "sync_pairs": [[str(p[0]), str(p[1])] for p in sync_pairs if isinstance(p, list) and len(p) == 2],
                    "resource_non_overlap": resource_pairs,
                },
                "G": {
                    "nodes": sorted(windows.keys()),
                    "edges": {
                        "precedence": [[str(p[0]), str(p[1])] for p in precedence if isinstance(p, list) and len(p) == 2],
                        "sync": [[str(p[0]), str(p[1])] for p in sync_pairs if isinstance(p, list) and len(p) == 2],
                        "resource_non_overlap": [[str(e["u"]), str(e["v"])] for e in resource_pairs],
                    },
                },
                "assignment": assignment,
                "windows": windows,
                "constraints": {
                    "precedence_pairs": [[str(p[0]), str(p[1])] for p in precedence if isinstance(p, list) and len(p) == 2],
                    "sync_pairs": [[str(p[0]), str(p[1])] for p in sync_pairs if isinstance(p, list) and len(p) == 2],
                    "resource_edges": [[str(e["u"]), str(e["v"])] for e in resource_pairs],
                },
                "dispatch_graph": {
                    "nodes": sorted(windows.keys()),
                    "edges": {
                        "precedence": [[str(p[0]), str(p[1])] for p in precedence if isinstance(p, list) and len(p) == 2],
                        "sync": [[str(p[0]), str(p[1])] for p in sync_pairs if isinstance(p, list) and len(p) == 2],
                        "resource_non_overlap": [[str(e["u"]), str(e["v"])] for e in resource_pairs],
                    },
                },
                "metrics": {
                    "flexibility_volume": float(flexibility_volume),
                    "critical_slack_margin": float(dispatch_comp["critical_slack_margin"]),
                    "robust_precedence_margin": float(dispatch_comp["robust_precedence_margin"]),
                    "sync_tolerance": float(dispatch_comp["sync_tolerance"]),
                    "resource_free_buffer": float(dispatch_comp["resource_free_buffer"]),
                    "dispatchability_success": bool(dispatch_comp["dispatchability_success"]),
                    "controllability_margin": float(dispatch_comp["controllability_margin"]),
                    "window_width_min": float(dispatch_comp["window_width_min"]),
                    "horizon": float(horizon),
                },
                "checks": {
                    "validity": validity,
                    "dispatchability": {
                        "dispatchable": bool(dispatch_comp["dispatchability_success"]),
                        "controllability_margin": float(dispatch_comp["controllability_margin"]),
                        "critical_slack_margin": float(dispatch_comp["critical_slack_margin"]),
                    },
                },
                "synthesis_params": {
                    "slack_expand_ratio": float(ratio),
                    "min_window_pad": int(pad),
                },
            }
            obj = envelope_objective(
                envelope,
                l_obj=float(l_obj),
                lambda_obj=float(lambda_obj),
                lambda_flex=float(lambda_flex),
                lambda_risk=float(lambda_risk),
            )
            envelope["objective"] = {
                "value": float(obj),
                "L_obj": float(l_obj),
                "lambda_obj": float(lambda_obj),
                "lambda_flex": float(lambda_flex),
                "lambda_risk": float(lambda_risk),
            }
            candidates.append(envelope)

    if not candidates:
        return {
            "x": {},
            "W": {},
            "R": {"precedence_pairs": [], "sync_pairs": [], "resource_non_overlap": []},
            "G": {"nodes": [], "edges": {"precedence": [], "sync": [], "resource_non_overlap": []}},
            "assignment": {},
            "windows": {},
            "constraints": {"precedence_pairs": [], "sync_pairs": [], "resource_edges": []},
            "dispatch_graph": {"nodes": [], "edges": {"precedence": [], "sync": [], "resource_non_overlap": []}},
            "metrics": {
                "flexibility_volume": 0.0,
                "critical_slack_margin": 0.0,
                "sync_tolerance": 0.0,
                "resource_free_buffer": 0.0,
                "dispatchability_success": False,
                "controllability_margin": -1.0,
                "window_width_min": 0.0,
                "horizon": 0.0,
            },
            "checks": {"validity": {"valid": False, "reasons": ["empty_envelope"]}},
            "objective": {"value": float("inf")},
            "synthesis_params": {"slack_expand_ratio": float(slack_expand_ratio), "min_window_pad": int(min_window_pad)},
        }

    valid_dispatchable = [
        c
        for c in candidates
        if bool(c.get("checks", {}).get("validity", {}).get("valid", False))
        and bool(c.get("checks", {}).get("dispatchability", {}).get("dispatchable", False))
    ]
    pool = valid_dispatchable if valid_dispatchable else candidates
    best = min(
        pool,
        key=lambda c: (
            float(c.get("objective", {}).get("value", float("inf"))),
            -float(c.get("metrics", {}).get("flexibility_volume", 0.0)),
        ),
    )
    best["synthesis_search"] = {
        "n_candidates": len(candidates),
        "n_valid_dispatchable": len(valid_dispatchable),
    }
    return best
