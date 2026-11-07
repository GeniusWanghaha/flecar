from __future__ import annotations

from collections import defaultdict, deque
from typing import Any


def _temporal_adjacency(reduced: dict[str, Any]) -> dict[str, set[str]]:
    adj: dict[str, set[str]] = defaultdict(set)
    for pair in reduced.get("precedence_pairs", []):
        if not isinstance(pair, list) or len(pair) != 2:
            continue
        u, v = str(pair[0]), str(pair[1])
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _sync_adjacency(reduced: dict[str, Any]) -> dict[str, set[str]]:
    adj: dict[str, set[str]] = defaultdict(set)
    if isinstance(reduced.get("sync_pairs"), list):
        for pair in reduced["sync_pairs"]:
            if not isinstance(pair, list) or len(pair) != 2:
                continue
            u, v = str(pair[0]), str(pair[1])
            adj[u].add(v)
            adj[v].add(u)
        if adj:
            return adj

    nodes = reduced.get("nodes", {})
    for edge in reduced.get("sync_invalid_edges", []):
        if not isinstance(edge, (list, tuple)) or len(edge) != 2:
            continue
        su = str(nodes.get(str(edge[0]), {}).get("subtask_id", ""))
        sv = str(nodes.get(str(edge[1]), {}).get("subtask_id", ""))
        if not su or not sv or su == sv:
            continue
        adj[su].add(sv)
        adj[sv].add(su)
    return adj


def _resource_adjacency(reduced: dict[str, Any]) -> dict[str, set[str]]:
    adj: dict[str, set[str]] = defaultdict(set)
    nodes = reduced.get("nodes", {})

    by_subtask: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for node in nodes.values():
        by_subtask[str(node["subtask_id"])].append(node)

    subtasks = sorted(by_subtask.keys())
    for i, s1 in enumerate(subtasks):
        rows1 = by_subtask[s1]
        res1 = set()
        agents1 = set()
        for c in rows1:
            res1.update(str(x) for x in c.get("resources", []))
            agents1.add(str(c.get("agent_id", "")))
        for s2 in subtasks[i + 1 :]:
            rows2 = by_subtask[s2]
            res2 = set()
            agents2 = set()
            for c in rows2:
                res2.update(str(x) for x in c.get("resources", []))
                agents2.add(str(c.get("agent_id", "")))
            if (res1 & res2) or (agents1 & agents2):
                adj[s1].add(s2)
                adj[s2].add(s1)

    for edge in reduced.get("conflict_edges", []):
        if not isinstance(edge, (list, tuple)) or len(edge) != 2:
            continue
        su = str(nodes.get(str(edge[0]), {}).get("subtask_id", ""))
        sv = str(nodes.get(str(edge[1]), {}).get("subtask_id", ""))
        if not su or not sv or su == sv:
            continue
        adj[su].add(sv)
        adj[sv].add(su)

    return adj


def _merge_adjacency(parts: list[dict[str, set[str]]]) -> dict[str, set[str]]:
    merged: dict[str, set[str]] = defaultdict(set)
    for part in parts:
        for sid, nxt in part.items():
            merged[sid].update(nxt)
            for x in nxt:
                merged[x].add(sid)
    return merged


def localize_controllability_region(
    reduced: dict[str, Any],
    diagnostics: dict[str, Any],
    *,
    radius: int = 1,
) -> set[str]:
    impacted = set(str(x) for x in diagnostics.get("impacted_subtasks", []))
    if not impacted:
        return set()

    temporal_adj = _temporal_adjacency(reduced)
    resource_adj = _resource_adjacency(reduced)
    sync_adj = _sync_adjacency(reduced)
    merged_adj = _merge_adjacency([temporal_adj, resource_adj, sync_adj])

    visited = set(impacted)
    q = deque((sid, 0) for sid in impacted)
    while q:
        sid, depth = q.popleft()
        if depth >= int(radius):
            continue
        for nxt in merged_adj.get(sid, set()):
            if nxt in visited:
                continue
            visited.add(nxt)
            q.append((nxt, depth + 1))

    if float(diagnostics.get("future_conflict_forecast", 0.0)) >= 0.5:
        for sid in list(visited):
            visited.update(merged_adj.get(sid, set()))

    if float(diagnostics.get("synchronized_task_risk", 0.0)) > 0.0:
        for sid in list(visited):
            visited.update(sync_adj.get(sid, set()))

    return visited
