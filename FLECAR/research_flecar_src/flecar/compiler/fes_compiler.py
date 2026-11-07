from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from typing import Any


@dataclass
class FESScoreWeights:
    alpha_b: float = 0.25
    alpha_c: float = 0.30
    alpha_s: float = 0.15
    alpha_d: float = 0.10
    alpha_f: float = 0.20
    alpha_q: float = 0.20


def _overlap(a_start: int, a_dur: int, b_start: int, b_dur: int) -> bool:
    ae = a_start + a_dur
    be = b_start + b_dur
    return not (ae <= b_start or be <= a_start)


def _pred_succ_map(precedence: list[list[str]]) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    pred: dict[str, set[str]] = defaultdict(set)
    succ: dict[str, set[str]] = defaultdict(set)
    for pair in precedence:
        if isinstance(pair, list) and len(pair) == 2:
            u = str(pair[0])
            v = str(pair[1])
            pred[v].add(u)
            succ[u].add(v)
            pred.setdefault(u, set())
            succ.setdefault(v, set())
    return pred, succ


def _sync_degree(instance: dict[str, Any]) -> dict[str, int]:
    deg: dict[str, int] = defaultdict(int)
    for item in instance.get("sync_constraints", []):
        sids = [str(x) for x in item.get("subtask_ids", [])]
        for i in range(len(sids)):
            for j in range(i + 1, len(sids)):
                deg[sids[i]] += 1
                deg[sids[j]] += 1
    return deg


def compute_fes_candidate_scores(instance: dict[str, Any], weights: FESScoreWeights) -> list[dict[str, Any]]:
    cands = instance.get("candidate_assignments", [])
    if not cands:
        return []

    by_subtask: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_agent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    res_freq: dict[str, int] = defaultdict(int)
    for c in cands:
        sid = str(c["subtask_id"])
        aid = str(c["agent_id"])
        by_subtask[sid].append(c)
        by_agent[aid].append(c)
        for r in c.get("resources", []):
            res_freq[str(r)] += 1

    total_res = max(1, sum(res_freq.values()))
    pred, succ = _pred_succ_map(instance.get("precedence", []))
    sync_deg = _sync_degree(instance)

    start_ranges: dict[str, tuple[int, int]] = {}
    for sid, rows in by_subtask.items():
        starts = [int(r["start"]) for r in rows]
        start_ranges[sid] = (min(starts), max(starts))

    avg_branching = sum(len(v) for v in by_subtask.values()) / max(1.0, float(len(by_subtask)))

    scores: list[dict[str, Any]] = []
    for c in cands:
        cid = str(c["candidate_id"])
        sid = str(c["subtask_id"])
        aid = str(c["agent_id"])
        start = int(c["start"])
        dur = int(c["duration"])
        cost = float(c["cost"])
        resources = [str(r) for r in c.get("resources", [])]

        base = cost

        bottleneck = sum(float(res_freq[r]) / float(total_res) for r in resources)

        conflict = 0.0
        overlap_total = 0.0
        for other in cands:
            if str(other["candidate_id"]) == cid or str(other["subtask_id"]) == sid:
                continue
            if _overlap(start, dur, int(other["start"]), int(other["duration"])):
                overlap_total += 1.0
                ores = {str(r) for r in other.get("resources", [])}
                same_agent = str(other.get("agent_id", "")) == aid
                if ores.intersection(resources) or same_agent:
                    conflict += 1.0

        slack_risk = 0.0
        for psid in pred.get(sid, set()):
            prows = by_subtask.get(psid, [])
            if not prows:
                continue
            best_pred_end = min(int(p["start"]) + int(p["duration"]) for p in prows)
            slack_risk += max(0.0, float(best_pred_end - start))

        reuse = float(len(by_agent.get(aid, []))) / float(max(1, len(cands)))

        lo, hi = start_ranges.get(sid, (start, start))
        span = max(1, hi - lo)
        center = (lo + hi) / 2.0
        branch = float(len(by_subtask.get(sid, []))) / max(1.0, avg_branching)

        # flex_gain(c): reward temporal room, branching alternatives, and short duration.
        flex_gain = (
            (1.0 - abs(start - center) / float(span))
            + min(1.5, branch)
            + (1.0 / max(1.0, float(dur)))
        )

        # ctrl_fragility(c): penalize precedence/sync centrality + conflict susceptibility.
        degree_fragility = float(len(pred.get(sid, set())) + len(succ.get(sid, set()))) * 0.25
        sync_fragility = float(sync_deg.get(sid, 0)) * 0.20
        overlap_fragility = float(conflict) / max(1.0, float(overlap_total)) if overlap_total > 0 else 0.0
        resource_fragility = float(len(resources) > 1) * 0.20
        ctrl_fragility = degree_fragility + sync_fragility + overlap_fragility + resource_fragility

        score = (
            base
            + weights.alpha_b * bottleneck
            + weights.alpha_c * conflict
            + weights.alpha_s * slack_risk
            + weights.alpha_d * reuse
            - weights.alpha_f * flex_gain
            + weights.alpha_q * ctrl_fragility
        )
        scores.append(
            {
                "candidate_id": cid,
                "subtask_id": sid,
                "score": float(score),
                "base": float(base),
                "bottleneck": float(bottleneck),
                "conflict": float(conflict),
                "slack_risk": float(slack_risk),
                "reuse": float(reuse),
                "flex_gain": float(flex_gain),
                "ctrl_fragility": float(ctrl_fragility),
            }
        )
    return scores


def compile_fes_candidates(
    instance: dict[str, Any],
    *,
    top_k: int,
    weights: FESScoreWeights,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    scores = compute_fes_candidate_scores(instance, weights)
    if not scores:
        return deepcopy(instance), []

    by_subtask_score: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for s in scores:
        by_subtask_score[str(s["subtask_id"])].append(s)

    keep_ids: set[str] = set()
    for _sid, rows in by_subtask_score.items():
        rows_sorted = sorted(rows, key=lambda r: (float(r["score"]), str(r["candidate_id"])))
        for r in rows_sorted[: max(1, int(top_k))]:
            keep_ids.add(str(r["candidate_id"]))

    compiled = deepcopy(instance)
    compiled["candidate_assignments"] = [
        c for c in instance.get("candidate_assignments", []) if str(c["candidate_id"]) in keep_ids
    ]
    compiled.setdefault("metadata", {})["compiler"] = "FES"
    compiled["metadata"]["compiler_top_k"] = int(top_k)
    compiled["metadata"]["compiler_weights"] = {
        "alpha_b": weights.alpha_b,
        "alpha_c": weights.alpha_c,
        "alpha_s": weights.alpha_s,
        "alpha_d": weights.alpha_d,
        "alpha_f": weights.alpha_f,
        "alpha_q": weights.alpha_q,
    }
    return compiled, scores


def compile_bacb_plus_candidates(instance: dict[str, Any], *, top_k: int) -> dict[str, Any]:
    """BACB+ stage in FLECAR: BACB retained as compiler sub-module, not headline method."""
    w = FESScoreWeights(alpha_b=0.25, alpha_c=0.30, alpha_s=0.15, alpha_d=0.10, alpha_f=0.0, alpha_q=0.0)
    compiled, _ = compile_fes_candidates(instance, top_k=top_k, weights=w)
    compiled.setdefault("metadata", {})["compiler"] = "BACB+"
    return compiled
