"""Greedy baseline solver on reduced scheduling graph."""

from __future__ import annotations

from time import perf_counter
from typing import Any

from qcma.solvers.common import is_feasible, normalize_edge, objective_from_selection


def solve_greedy(reduced: dict[str, Any]) -> dict[str, Any]:
    t0 = perf_counter()
    nodes = reduced["nodes"]
    groups = reduced["unique_groups"]

    bad_edges = set()
    for key in ("conflict_edges", "precedence_invalid_edges", "sync_invalid_edges"):
        for u, v in reduced.get(key, []):
            bad_edges.add(normalize_edge(u, v))

    selected: list[str] = []
    selected_set: set[str] = set()

    subtasks = sorted(groups.keys())
    for sid in subtasks:
        cands = sorted(
            groups[sid], key=lambda cid: (nodes[cid]["cost"], nodes[cid]["start"], cid)
        )
        chosen = None
        for cid in cands:
            ok = True
            for other in selected_set:
                if normalize_edge(cid, other) in bad_edges:
                    ok = False
                    break
            if ok:
                chosen = cid
                break
        if chosen is None:
            continue
        selected.append(chosen)
        selected_set.add(chosen)

    runtime = perf_counter() - t0
    feasible = is_feasible(reduced, selected)
    objective = objective_from_selection(reduced, selected) if selected else float("inf")
    return {
        "solver": "greedy",
        "selected_candidate_ids": selected,
        "feasible": feasible,
        "objective": objective,
        "runtime_sec": runtime,
    }

