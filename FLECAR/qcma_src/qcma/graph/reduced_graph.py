"""Build reduced scheduling graph from unified QCMA instance."""

from __future__ import annotations

import random
from collections import defaultdict
from itertools import combinations
from typing import Any


def _overlap(a_start: int, a_dur: int, b_start: int, b_dur: int) -> bool:
    a_end = a_start + a_dur
    b_end = b_start + b_dur
    return not (a_end <= b_start or b_end <= a_start)


def _edge_key(u: str, v: str) -> tuple[str, str]:
    return (u, v) if u < v else (v, u)


def _candidate_temporal_risk(
    cand: dict[str, Any],
    all_candidates: list[dict[str, Any]],
) -> int:
    """Approximate risk: same-agent temporal overlaps with other subtasks."""
    s = int(cand["start"])
    d = int(cand["duration"])
    a = str(cand["agent_id"])
    sid = str(cand["subtask_id"])
    risk = 0
    for other in all_candidates:
        if str(other["subtask_id"]) == sid:
            continue
        if str(other["agent_id"]) != a:
            continue
        if _overlap(s, d, int(other["start"]), int(other["duration"])):
            risk += 1
    return risk


def _build_precedence_maps(
    precedence_pairs: list[list[str]],
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    pred_map: dict[str, set[str]] = defaultdict(set)
    succ_map: dict[str, set[str]] = defaultdict(set)
    for pair in precedence_pairs:
        if not isinstance(pair, list) or len(pair) != 2:
            continue
        u = str(pair[0])
        v = str(pair[1])
        succ_map[u].add(v)
        pred_map[v].add(u)
        pred_map.setdefault(u, set())
        succ_map.setdefault(v, set())
    return pred_map, succ_map


def _dominance_prune_candidates(
    candidates: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Semantics-preserving dominated-candidate pruning.

    Rule (within same subtask and same agent):
    candidate j is dominated by i if:
      start_i <= start_j, duration_i <= duration_j, cost_i <= cost_j,
    and at least one strict inequality.
    """
    by_group: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    by_subtask: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in candidates:
        sid = str(c["subtask_id"])
        aid = str(c["agent_id"])
        by_group[(sid, aid)].append(c)
        by_subtask[sid].append(c)

    dropped: set[str] = set()
    for (_sid, _aid), rows in by_group.items():
        for i in range(len(rows)):
            ci = rows[i]
            if str(ci["candidate_id"]) in dropped:
                continue
            si = int(ci["start"])
            di = int(ci["duration"])
            co_i = float(ci["cost"])
            for j in range(len(rows)):
                if i == j:
                    continue
                cj = rows[j]
                cjid = str(cj["candidate_id"])
                if cjid in dropped:
                    continue
                sj = int(cj["start"])
                dj = int(cj["duration"])
                co_j = float(cj["cost"])
                dominated = (si <= sj) and (di <= dj) and (co_i <= co_j)
                strictly_better = (si < sj) or (di < dj) or (co_i < co_j)
                if dominated and strictly_better:
                    dropped.add(cjid)

    # Safety: never remove all candidates for a subtask.
    kept = [c for c in candidates if str(c["candidate_id"]) not in dropped]
    kept_by_sid: dict[str, int] = defaultdict(int)
    for c in kept:
        kept_by_sid[str(c["subtask_id"])] += 1
    for sid, rows in by_subtask.items():
        if kept_by_sid.get(sid, 0) > 0:
            continue
        best = min(rows, key=lambda x: (float(x["cost"]), int(x["start"]), int(x["duration"])))
        dropped.discard(str(best["candidate_id"]))

    kept = [c for c in candidates if str(c["candidate_id"]) not in dropped]
    return kept, sorted(dropped)


def _precedence_slack_penalty(
    cand: dict[str, Any],
    by_subtask: dict[str, list[dict[str, Any]]],
    pred_map: dict[str, set[str]],
) -> tuple[float, float]:
    """Return precedence violation indicator and negative slack penalty."""
    sid = str(cand["subtask_id"])
    preds = pred_map.get(sid, set())
    if not preds:
        return 0.0, 0.0

    pred_best_finish = 0
    for psid in preds:
        prow = by_subtask.get(psid, [])
        if not prow:
            continue
        best_pred = min(prow, key=lambda x: (float(x["cost"]), int(x["start"]), int(x["duration"])))
        pred_best_finish = max(pred_best_finish, int(best_pred["start"]) + int(best_pred["duration"]))

    start = int(cand["start"])
    slack = float(start - pred_best_finish)
    return (1.0 if slack < 0 else 0.0), float(max(0.0, -slack))


def _candidate_conflict_risk(
    cand: dict[str, Any],
    all_candidates: list[dict[str, Any]],
) -> int:
    """Approximate risk: temporal overlaps with shared resources/zones."""
    sid = str(cand["subtask_id"])
    s = int(cand["start"])
    d = int(cand["duration"])
    res = set(cand.get("resources", []))
    zone = str(cand.get("zone", ""))
    risk = 0
    for other in all_candidates:
        if str(other["subtask_id"]) == sid:
            continue
        if not _overlap(s, d, int(other["start"]), int(other["duration"])):
            continue
        ores = set(other.get("resources", []))
        ozone = str(other.get("zone", ""))
        if (res and ores and (res & ores)) or (zone and ozone and zone == ozone):
            risk += 1
    return risk


def _resource_bottleneck_index(candidates: list[dict[str, Any]]) -> dict[str, float]:
    """Estimate global resource bottleneck pressure from candidate pool."""
    freq: dict[str, int] = defaultdict(int)
    total = 0
    for c in candidates:
        for r in c.get("resources", []):
            key = str(r)
            freq[key] += 1
            total += 1
    if total <= 0:
        return {}
    return {k: float(v) / float(total) for k, v in freq.items()}


def _candidate_bottleneck_score(
    cand: dict[str, Any],
    bottleneck_index: dict[str, float],
) -> float:
    """Higher score indicates candidate uses globally congested resources."""
    if not bottleneck_index:
        return 0.0
    score = 0.0
    for r in cand.get("resources", []):
        score += float(bottleneck_index.get(str(r), 0.0))
    return score


def _subtask_hardness_score(
    sid: str,
    rows: list[dict[str, Any]],
    *,
    pred_map: dict[str, set[str]],
    succ_map: dict[str, set[str]],
    all_candidates: list[dict[str, Any]],
    bottleneck_index: dict[str, float],
) -> float:
    branching = float(len(rows))
    dep = float(len(pred_map.get(sid, set())) + len(succ_map.get(sid, set())))
    if rows:
        conflict = sum(_candidate_conflict_risk(c, all_candidates) for c in rows) / float(len(rows))
        bottleneck = (
            sum(_candidate_bottleneck_score(c, bottleneck_index) for c in rows) / float(len(rows))
        )
    else:
        conflict = 0.0
        bottleneck = 0.0
    return 0.45 * branching + 0.35 * dep + 0.2 * conflict + 0.3 * bottleneck


def _adaptive_hardness_top_k_candidates(
    candidates: list[dict[str, Any]],
    top_k: int,
    precedence_pairs: list[list[str]],
) -> tuple[dict[str, dict[str, Any]], dict[str, list[str]], dict[str, int]]:
    """Module A1: hardness-calibrated adaptive top-k candidate compiler."""
    by_subtask: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in candidates:
        by_subtask[str(c["subtask_id"])].append(c)
    pred_map, succ_map = _build_precedence_maps(precedence_pairs)
    bottleneck_index = _resource_bottleneck_index(candidates)

    hardness_by_sid: dict[str, float] = {}
    for sid, rows in by_subtask.items():
        hardness_by_sid[sid] = _subtask_hardness_score(
            sid,
            rows,
            pred_map=pred_map,
            succ_map=succ_map,
            all_candidates=candidates,
            bottleneck_index=bottleneck_index,
        )

    if hardness_by_sid:
        hs = list(hardness_by_sid.values())
        lo = min(hs)
        hi = max(hs)
    else:
        lo = 0.0
        hi = 0.0

    def _hard_norm(v: float) -> float:
        if hi <= lo + 1e-12:
            return 0.5
        return (v - lo) / (hi - lo)

    kept: dict[str, dict[str, Any]] = {}
    by_subtask_ids: dict[str, list[str]] = {}
    adaptive_topk: dict[str, int] = {}
    for sid, rows in by_subtask.items():
        h = _hard_norm(hardness_by_sid.get(sid, 0.0))
        # Hard subtasks keep more alternatives; easy subtasks can be tightened.
        if h >= 0.65:
            sid_k = min(len(rows), top_k + 1)
        elif h <= 0.25:
            sid_k = min(len(rows), max(1, top_k - 1))
        else:
            sid_k = min(len(rows), top_k)
        sid_k = max(1, sid_k)
        adaptive_topk[sid] = sid_k

        def _score(x: dict[str, Any]) -> tuple[float, ...]:
            temporal_risk = _candidate_temporal_risk(x, candidates)
            conflict_risk = _candidate_conflict_risk(x, candidates)
            prec_viol, neg_slack = _precedence_slack_penalty(x, by_subtask, pred_map)
            bottleneck = _candidate_bottleneck_score(x, bottleneck_index)
            score = (
                float(x["cost"])
                + 1.20 * prec_viol
                + 0.18 * float(temporal_risk)
                + 0.30 * float(conflict_risk)
                + 0.12 * float(neg_slack)
                + 0.22 * float(bottleneck)
            )
            return (score, int(x["start"]), int(x["duration"]))

        rows_sorted = sorted(rows, key=_score)
        selected = rows_sorted[:sid_k]
        by_subtask_ids[sid] = []
        for c in selected:
            cid = str(c["candidate_id"])
            kept[cid] = c
            by_subtask_ids[sid].append(cid)
    return kept, by_subtask_ids, adaptive_topk


def _bottleneck_diverse_top_k_candidates(
    candidates: list[dict[str, Any]],
    top_k: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, list[str]], dict[str, int]]:
    """Module A2: bottleneck-aware candidate bundling with diversity control."""
    by_subtask: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in candidates:
        by_subtask[str(c["subtask_id"])].append(c)
    bottleneck_index = _resource_bottleneck_index(candidates)

    kept: dict[str, dict[str, Any]] = {}
    by_subtask_ids: dict[str, list[str]] = {}
    adaptive_topk: dict[str, int] = {}
    for sid, rows in by_subtask.items():
        sid_k = min(len(rows), max(1, top_k))
        adaptive_topk[sid] = sid_k
        rows_sorted = sorted(
            rows,
            key=lambda x: (
                float(x["cost"]) + 0.25 * _candidate_bottleneck_score(x, bottleneck_index),
                int(x["start"]),
                int(x["duration"]),
            ),
        )
        selected: list[dict[str, Any]] = []
        used_agents: set[str] = set()
        used_resources: set[str] = set()
        while len(selected) < sid_k and rows_sorted:
            best_idx = 0
            best_val: tuple[float, ...] | None = None
            for i, cand in enumerate(rows_sorted):
                agent = str(cand.get("agent_id", ""))
                resources = {str(r) for r in cand.get("resources", [])}
                # Prefer diverse bundles first, then low weighted cost.
                diversity_pen = 0.0
                if agent and agent in used_agents:
                    diversity_pen += 1.0
                if resources and (resources & used_resources):
                    diversity_pen += 1.0
                val = (
                    diversity_pen,
                    float(cand["cost"]) + 0.30 * _candidate_bottleneck_score(cand, bottleneck_index),
                    int(cand["start"]),
                    int(cand["duration"]),
                )
                if best_val is None or val < best_val:
                    best_val = val
                    best_idx = i
            chosen = rows_sorted.pop(best_idx)
            selected.append(chosen)
            used_agents.add(str(chosen.get("agent_id", "")))
            used_resources.update({str(r) for r in chosen.get("resources", [])})

        by_subtask_ids[sid] = []
        for c in selected:
            cid = str(c["candidate_id"])
            kept[cid] = c
            by_subtask_ids[sid].append(cid)
    return kept, by_subtask_ids, adaptive_topk


def _adaptive_top_k_candidates(
    candidates: list[dict[str, Any]],
    top_k: int,
    precedence_pairs: list[list[str]],
) -> tuple[dict[str, dict[str, Any]], dict[str, list[str]], dict[str, int]]:
    by_subtask: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in candidates:
        by_subtask[str(c["subtask_id"])].append(c)
    pred_map, succ_map = _build_precedence_maps(precedence_pairs)

    n_subtasks = max(1, len(by_subtask))
    branching = len(candidates) / n_subtasks
    # Keep cost as the dominant term to preserve baseline behavior, then
    # apply light adaptive regularization for conflict/slack robustness.
    conflict_weight = 0.35 if branching >= 6.0 else 0.22
    temporal_weight = 0.20 if branching >= 6.0 else 0.12
    slack_weight = 0.10 if branching >= 5.0 else 0.06
    precedence_violation_penalty = 1.25

    kept: dict[str, dict[str, Any]] = {}
    by_subtask_ids: dict[str, list[str]] = {}
    adaptive_topk: dict[str, int] = {}

    for sid, rows in by_subtask.items():
        in_deg = len(pred_map.get(sid, set()))
        out_deg = len(succ_map.get(sid, set()))
        # Adaptive top-k policy: keep more alternatives for structurally central subtasks.
        if in_deg + out_deg >= 2:
            sid_k = min(len(rows), top_k + 1)
        elif branching <= 3.0:
            sid_k = min(len(rows), max(1, top_k - 1))
        else:
            sid_k = min(len(rows), top_k)
        sid_k = max(1, sid_k)
        adaptive_topk[sid] = sid_k

        def _score(x: dict[str, Any]) -> tuple[float, ...]:
            temporal_risk = _candidate_temporal_risk(x, candidates)
            conflict_risk = _candidate_conflict_risk(x, candidates)
            prec_viol, neg_slack = _precedence_slack_penalty(x, by_subtask, pred_map)
            score = (
                float(x["cost"])
                + precedence_violation_penalty * prec_viol
                + temporal_weight * float(temporal_risk)
                + conflict_weight * float(conflict_risk)
                + slack_weight * float(neg_slack)
            )
            return (
                score,
                int(x["start"]),
                int(x["duration"]),
            )

        rows_sorted = sorted(rows, key=_score)
        selected = rows_sorted[:sid_k]
        by_subtask_ids[sid] = []
        for c in selected:
            cid = str(c["candidate_id"])
            kept[cid] = c
            by_subtask_ids[sid].append(cid)
    return kept, by_subtask_ids, adaptive_topk


def _top_k_candidates(
    candidates: list[dict[str, Any]],
    top_k: int,
    *,
    candidate_rank_mode: str = "cost_start_duration",
) -> tuple[dict[str, dict[str, Any]], dict[str, list[str]]]:
    by_subtask: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in candidates:
        by_subtask[c["subtask_id"]].append(c)

    kept: dict[str, dict[str, Any]] = {}
    by_subtask_ids: dict[str, list[str]] = {}
    for sid, rows in by_subtask.items():
        rank_mode = str(candidate_rank_mode).strip().lower()
        if rank_mode == "risk_aware":
            rows_sorted = sorted(
                rows,
                key=lambda x: (
                    _candidate_temporal_risk(x, candidates),
                    x["cost"],
                    x["start"],
                    x["duration"],
                ),
            )
            selected = rows_sorted[:top_k]
        elif rank_mode == "diverse_time":
            # Keep low-cost candidates while preserving temporal diversity to improve
            # feasibility search flexibility under tight budgets.
            rows_sorted = sorted(rows, key=lambda x: (x["cost"], x["start"], x["duration"]))
            if not rows_sorted:
                selected = []
            else:
                selected = [rows_sorted[0]]
                remaining = rows_sorted[1:]
                while len(selected) < top_k and remaining:
                    best_idx = 0
                    best_score = None
                    for i, cand in enumerate(remaining):
                        c_start = int(cand["start"])
                        min_dist = min(abs(c_start - int(s["start"])) for s in selected)
                        # Prefer greater temporal spread, but keep costs bounded.
                        score = (min_dist, -float(cand["cost"]), -int(cand["duration"]))
                        if best_score is None or score > best_score:
                            best_score = score
                            best_idx = i
                    selected.append(remaining.pop(best_idx))
                # Final deterministic order.
                selected = sorted(selected, key=lambda x: (x["cost"], x["start"], x["duration"]))
        else:
            rows_sorted = sorted(rows, key=lambda x: (x["cost"], x["start"], x["duration"]))
            selected = rows_sorted[:top_k]
        by_subtask_ids[sid] = []
        for c in selected:
            cid = c["candidate_id"]
            kept[cid] = c
            by_subtask_ids[sid].append(cid)
    return kept, by_subtask_ids


def _scale_or_perturb_edges(
    edges: set[tuple[str, str]],
    all_candidate_ids: list[str],
    subtask_by_cid: dict[str, str],
    *,
    density_scale: float,
    rng_seed: int,
) -> set[tuple[str, str]]:
    if density_scale <= 0:
        return set()
    if abs(density_scale - 1.0) < 1e-12:
        return set(edges)

    rng = random.Random(rng_seed)
    edge_list = sorted(edges)

    if density_scale < 1.0:
        keep = max(0, int(round(len(edge_list) * density_scale)))
        rng.shuffle(edge_list)
        return set(edge_list[:keep])

    # density_scale > 1.0: add synthetic conflict edges across different subtasks.
    out = set(edge_list)
    target = int(round(len(edge_list) * density_scale))
    if target <= len(out):
        return out

    cids = list(all_candidate_ids)
    trials = 0
    max_trials = max(1000, 20 * len(cids))
    while len(out) < target and trials < max_trials:
        trials += 1
        u = cids[rng.randrange(len(cids))]
        v = cids[rng.randrange(len(cids))]
        if u == v:
            continue
        if subtask_by_cid.get(u) == subtask_by_cid.get(v):
            continue
        out.add(_edge_key(u, v))
    return out


def build_reduced_graph(
    instance: dict[str, Any],
    *,
    top_k: int = 3,
    constraint_density_scale: float = 1.0,
    density_seed: int = 7,
    candidate_rank_mode: str = "cost_start_duration",
    enable_dominated_pruning: bool | None = None,
) -> dict[str, Any]:
    all_candidates_raw = instance["candidate_assignments"]
    precedence_pairs = instance.get("precedence", [])
    rank_mode = str(candidate_rank_mode).strip().lower()

    if enable_dominated_pruning is None:
        enable_dominated_pruning = rank_mode in {"adaptive_risk_slack", "adaptive_hardness_topk"}

    all_candidates = list(all_candidates_raw)
    dropped_dominated: list[str] = []
    if enable_dominated_pruning:
        all_candidates, dropped_dominated = _dominance_prune_candidates(all_candidates)

    adaptive_topk: dict[str, int] = {}
    if rank_mode == "adaptive_risk_slack":
        nodes, subtask_candidates, adaptive_topk = _adaptive_top_k_candidates(
            all_candidates,
            top_k=top_k,
            precedence_pairs=precedence_pairs,
        )
    elif rank_mode == "adaptive_hardness_topk":
        nodes, subtask_candidates, adaptive_topk = _adaptive_hardness_top_k_candidates(
            all_candidates,
            top_k=top_k,
            precedence_pairs=precedence_pairs,
        )
    elif rank_mode == "bottleneck_diverse":
        nodes, subtask_candidates, adaptive_topk = _bottleneck_diverse_top_k_candidates(
            all_candidates,
            top_k=top_k,
        )
    else:
        nodes, subtask_candidates = _top_k_candidates(
            all_candidates,
            top_k=top_k,
            candidate_rank_mode=rank_mode,
        )

    conflict_edges: set[tuple[str, str]] = set()
    precedence_invalid_edges: set[tuple[str, str]] = set()
    sync_invalid_edges: set[tuple[str, str]] = set()

    # Conflict: same agent overlapping times
    node_list = list(nodes.values())
    for c1, c2 in combinations(node_list, 2):
        if c1["agent_id"] == c2["agent_id"] and _overlap(
            c1["start"], c1["duration"], c2["start"], c2["duration"]
        ):
            conflict_edges.add(_edge_key(c1["candidate_id"], c2["candidate_id"]))

    # Conflict: resource and spatial constraints
    subtask_to_candidates = subtask_candidates
    for rc in instance.get("resource_conflicts", []):
        sids = rc.get("subtask_ids", [])
        for i in range(len(sids)):
            for j in range(i + 1, len(sids)):
                for c1 in subtask_to_candidates.get(sids[i], []):
                    for c2 in subtask_to_candidates.get(sids[j], []):
                        n1, n2 = nodes[c1], nodes[c2]
                        if _overlap(
                            n1["start"], n1["duration"], n2["start"], n2["duration"]
                        ):
                            conflict_edges.add(_edge_key(c1, c2))

    for sc in instance.get("spatial_conflicts", []):
        sids = sc.get("subtask_ids", [])
        for i in range(len(sids)):
            for j in range(i + 1, len(sids)):
                for c1 in subtask_to_candidates.get(sids[i], []):
                    for c2 in subtask_to_candidates.get(sids[j], []):
                        n1, n2 = nodes[c1], nodes[c2]
                        if _overlap(
                            n1["start"], n1["duration"], n2["start"], n2["duration"]
                        ):
                            conflict_edges.add(_edge_key(c1, c2))

    # Precedence invalid pair edges
    for u_sid, v_sid in instance.get("precedence", []):
        for u_cid in subtask_to_candidates.get(u_sid, []):
            for v_cid in subtask_to_candidates.get(v_sid, []):
                u_node = nodes[u_cid]
                v_node = nodes[v_cid]
                if u_node["start"] + u_node["duration"] > v_node["start"]:
                    precedence_invalid_edges.add(_edge_key(u_cid, v_cid))

    # Sync invalid pair edges
    for sync in instance.get("sync_constraints", []):
        sids = sync.get("subtask_ids", [])
        max_lag = int(sync.get("max_lag", 0))
        for i in range(len(sids)):
            for j in range(i + 1, len(sids)):
                sid_i, sid_j = sids[i], sids[j]
                for c1 in subtask_to_candidates.get(sid_i, []):
                    for c2 in subtask_to_candidates.get(sid_j, []):
                        lag = abs(nodes[c1]["start"] - nodes[c2]["start"])
                        if lag > max_lag:
                            sync_invalid_edges.add(_edge_key(c1, c2))

    subtask_by_cid = {cid: nodes[cid]["subtask_id"] for cid in nodes}
    scaled_conflict = _scale_or_perturb_edges(
        conflict_edges,
        list(nodes.keys()),
        subtask_by_cid,
        density_scale=constraint_density_scale,
        rng_seed=density_seed,
    )
    scaled_prec = _scale_or_perturb_edges(
        precedence_invalid_edges,
        list(nodes.keys()),
        subtask_by_cid,
        density_scale=constraint_density_scale,
        rng_seed=density_seed + 17,
    )
    scaled_sync = _scale_or_perturb_edges(
        sync_invalid_edges,
        list(nodes.keys()),
        subtask_by_cid,
        density_scale=constraint_density_scale,
        rng_seed=density_seed + 31,
    )

    reduced = {
        "episode_id": instance["episode_id"],
        "task_text": instance["task_text"],
        "top_k": top_k,
        "constraint_density_scale": constraint_density_scale,
        "precedence_pairs": precedence_pairs,
        "nodes": nodes,
        "subtask_candidates": subtask_candidates,
        "unique_groups": subtask_candidates,
        "conflict_edges": sorted(list(scaled_conflict)),
        "precedence_invalid_edges": sorted(list(scaled_prec)),
        "sync_invalid_edges": sorted(list(scaled_sync)),
        "metadata": {
            "n_subtasks": len(instance["subtasks"]),
            "n_candidates_raw": len(all_candidates_raw),
            "n_candidates_after_dominated_pruning": len(all_candidates),
            "n_nodes_reduced": len(nodes),
            "n_conflict_edges": len(scaled_conflict),
            "n_precedence_invalid_edges": len(scaled_prec),
            "n_sync_invalid_edges": len(scaled_sync),
            "candidate_rank_mode": rank_mode,
            "dominated_pruning_enabled": bool(enable_dominated_pruning),
            "n_dominated_candidates_dropped": len(dropped_dominated),
            "adaptive_topk_by_subtask": adaptive_topk,
        },
    }
    return reduced


def reduced_graph_stats(reduced_graph: dict[str, Any]) -> dict[str, int]:
    md = reduced_graph["metadata"]
    return {
        "n_subtasks": int(md["n_subtasks"]),
        "n_candidates_raw": int(md["n_candidates_raw"]),
        "n_nodes_reduced": int(md["n_nodes_reduced"]),
        "n_conflict_edges": int(md["n_conflict_edges"]),
        "n_precedence_invalid_edges": int(md["n_precedence_invalid_edges"]),
        "n_sync_invalid_edges": int(md["n_sync_invalid_edges"]),
    }
