"""Beam-search baseline solver on reduced scheduling graph."""

from __future__ import annotations

from time import perf_counter
from typing import Any

from qcma.solvers.common import (
    is_feasible,
    normalize_edge,
    objective_from_selection,
    violation_counts,
)


def _partial_score(
    reduced: dict[str, Any], selected: list[str], selected_set: set[str]
) -> tuple[int, float]:
    vio = violation_counts(reduced, selected)
    total_v = sum(vio.values())
    cost = objective_from_selection(reduced, selected) if selected else 0.0
    return total_v, cost


def solve_beam(
    reduced: dict[str, Any], *, beam_width: int = 8, time_limit_sec: float = 1.0
) -> dict[str, Any]:
    t0 = perf_counter()
    nodes = reduced["nodes"]
    groups = reduced["unique_groups"]
    subtasks = sorted(groups.keys())

    bad_edges = set()
    for key in ("conflict_edges", "precedence_invalid_edges", "sync_invalid_edges"):
        for u, v in reduced.get(key, []):
            bad_edges.add(normalize_edge(u, v))

    beams: list[tuple[list[str], set[str]]] = [([], set())]

    for sid in subtasks:
        new_beams: list[tuple[list[str], set[str]]] = []
        cands = sorted(
            groups[sid], key=lambda cid: (nodes[cid]["cost"], nodes[cid]["start"], cid)
        )
        for selected, selected_set in beams:
            for cid in cands:
                ok = True
                for other in selected_set:
                    if normalize_edge(cid, other) in bad_edges:
                        ok = False
                        break
                if not ok:
                    continue
                new_sel = selected + [cid]
                new_set = set(selected_set)
                new_set.add(cid)
                new_beams.append((new_sel, new_set))

        if not new_beams:
            break
        new_beams.sort(key=lambda b: _partial_score(reduced, b[0], b[1]))
        beams = new_beams[:beam_width]

        if perf_counter() - t0 > time_limit_sec:
            break

    if beams:
        best = min(beams, key=lambda b: _partial_score(reduced, b[0], b[1]))
        selected = best[0]
    else:
        selected = []

    runtime = perf_counter() - t0
    feasible = is_feasible(reduced, selected)
    objective = objective_from_selection(reduced, selected) if selected else float("inf")
    return {
        "solver": "beam",
        "selected_candidate_ids": selected,
        "feasible": feasible,
        "objective": objective,
        "runtime_sec": runtime,
        "beam_width": beam_width,
    }

