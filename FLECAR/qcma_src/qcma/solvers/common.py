"""Shared solver utilities and feasibility checks."""

from __future__ import annotations

from typing import Any


def edge_set(reduced: dict[str, Any], key: str) -> set[tuple[str, str]]:
    return {tuple(e) for e in reduced.get(key, [])}


def normalize_edge(u: str, v: str) -> tuple[str, str]:
    return (u, v) if u < v else (v, u)


def objective_from_selection(
    reduced: dict[str, Any], selected_candidate_ids: list[str]
) -> float:
    nodes = reduced["nodes"]
    return float(sum(float(nodes[cid]["cost"]) for cid in selected_candidate_ids))


def violation_counts(
    reduced: dict[str, Any], selected_candidate_ids: list[str]
) -> dict[str, int]:
    selected = set(selected_candidate_ids)
    violations = {
        "unique_assignment": 0,
        "conflict": 0,
        "precedence": 0,
        "sync": 0,
    }

    groups = reduced["unique_groups"]
    for _, cands in groups.items():
        n = sum(1 for cid in cands if cid in selected)
        if n != 1:
            violations["unique_assignment"] += abs(n - 1)

    for key, tag in [
        ("conflict_edges", "conflict"),
        ("precedence_invalid_edges", "precedence"),
        ("sync_invalid_edges", "sync"),
    ]:
        for u, v in reduced.get(key, []):
            if u in selected and v in selected:
                violations[tag] += 1
    return violations


def is_feasible(reduced: dict[str, Any], selected_candidate_ids: list[str]) -> bool:
    v = violation_counts(reduced, selected_candidate_ids)
    return sum(v.values()) == 0


def complete_selection(reduced: dict[str, Any], selected_candidate_ids: list[str]) -> bool:
    groups = reduced["unique_groups"]
    selected = set(selected_candidate_ids)
    return all(sum(1 for cid in cands if cid in selected) == 1 for cands in groups.values())

