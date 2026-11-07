"""Classical local-search baseline."""

from __future__ import annotations

import random
from time import perf_counter
from typing import Any

from qcma.solvers.common import objective_from_selection, violation_counts
from qcma.solvers.greedy import solve_greedy


def _score(reduced: dict[str, Any], selected: list[str]) -> tuple[int, float]:
    v = violation_counts(reduced, selected)
    n_vio = sum(v.values())
    cost = objective_from_selection(reduced, selected) if selected else float("inf")
    return n_vio, cost


def _replace_for_subtask(
    selected: list[str],
    sid: str,
    new_cid: str,
    groups: dict[str, list[str]],
) -> list[str]:
    out = [cid for cid in selected if cid not in groups[sid]]
    out.append(new_cid)
    return out


def solve_local_search(
    reduced: dict[str, Any], *, time_limit_sec: float = 1.0, seed: int = 0
) -> dict[str, Any]:
    t0 = perf_counter()
    rng = random.Random(seed)

    groups = reduced["unique_groups"]
    nodes = reduced["nodes"]

    greedy = solve_greedy(reduced)
    selected = list(greedy["selected_candidate_ids"])

    # Ensure one candidate per subtask as a starting point.
    selected_set = set(selected)
    for sid, cands in groups.items():
        chosen = [c for c in cands if c in selected_set]
        if not chosen and cands:
            selected.append(min(cands, key=lambda cid: (nodes[cid]["cost"], cid)))

    best = list(selected)
    best_score = _score(reduced, best)

    subtasks = sorted(groups.keys())
    while perf_counter() - t0 < time_limit_sec:
        sid = subtasks[rng.randrange(len(subtasks))]
        cands = groups[sid]
        if not cands:
            continue
        cid = cands[rng.randrange(len(cands))]
        trial = _replace_for_subtask(best, sid, cid, groups)
        trial_score = _score(reduced, trial)
        if trial_score < best_score:
            best, best_score = trial, trial_score

    runtime = perf_counter() - t0
    feasible = best_score[0] == 0
    objective = best_score[1]
    return {
        "solver": "local_search",
        "selected_candidate_ids": best,
        "feasible": feasible,
        "objective": objective,
        "runtime_sec": runtime,
    }

