"""Compile reduced scheduling graph into QUBO representation."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from qcma.compiler.penalty_config import PenaltyConfig


def _add_linear(linear: dict[int, float], i: int, v: float) -> None:
    linear[i] += float(v)


def _add_quadratic(quad: dict[tuple[int, int], float], i: int, j: int, v: float) -> None:
    if i == j:
        raise ValueError("Quadratic term requires i != j; use linear for x_i^2 via binary reduction.")
    key = (i, j) if i < j else (j, i)
    quad[key] += float(v)


def compile_qubo(
    reduced: dict[str, Any],
    *,
    penalties: PenaltyConfig | None = None,
    prev_selection: set[str] | None = None,
) -> dict[str, Any]:
    """Compile reduced graph into sparse QUBO coefficients."""
    if penalties is None:
        penalties = PenaltyConfig()
    if prev_selection is None:
        prev_selection = set()

    variables = sorted(reduced["nodes"].keys())
    var_index = {cid: i for i, cid in enumerate(variables)}
    group_indices = {
        sid: [var_index[cid] for cid in cids]
        for sid, cids in reduced["unique_groups"].items()
    }
    linear: dict[int, float] = defaultdict(float)
    quad: dict[tuple[int, int], float] = defaultdict(float)
    constant = 0.0

    # Soft cost term: lambda_cost * cost_c * x_c
    for cid in variables:
        i = var_index[cid]
        c = float(reduced["nodes"][cid]["cost"])
        _add_linear(linear, i, penalties.lambda_cost * c)

    # Unique assignment one-hot per subtask: p_assign * (sum x - 1)^2
    for _, group in reduced["unique_groups"].items():
        if not group:
            continue
        constant += penalties.p_assign
        for cid in group:
            i = var_index[cid]
            _add_linear(linear, i, -penalties.p_assign)
        for a in range(len(group)):
            for b in range(a + 1, len(group)):
                i = var_index[group[a]]
                j = var_index[group[b]]
                _add_quadratic(quad, i, j, 2.0 * penalties.p_assign)

    # Pairwise invalid edges: p * x_u x_v
    for key, weight in (
        ("conflict_edges", penalties.p_conflict),
        ("precedence_invalid_edges", penalties.p_precedence),
        ("sync_invalid_edges", penalties.p_sync),
    ):
        for u, v in reduced.get(key, []):
            i = var_index[u]
            j = var_index[v]
            _add_quadratic(quad, i, j, weight)

    # Repair linear term against previous plan (weighted Hamming)
    if penalties.lambda_repair > 0:
        lam = penalties.lambda_repair
        for cid in variables:
            i = var_index[cid]
            if cid in prev_selection:
                # |x-1| = 1 - x
                constant += lam
                _add_linear(linear, i, -lam)
            else:
                # |x-0| = x
                _add_linear(linear, i, lam)

    # Export sparse format + metadata map for traceability
    linear_terms = [{"i": i, "value": v} for i, v in sorted(linear.items()) if abs(v) > 1e-12]
    quadratic_terms = [
        {"i": i, "j": j, "value": v}
        for (i, j), v in sorted(quad.items())
        if abs(v) > 1e-12
    ]

    return {
        "variables": variables,
        "var_index": var_index,
        "group_indices": group_indices,
        "linear_terms": linear_terms,
        "quadratic_terms": quadratic_terms,
        "constant": float(constant),
        "penalties": penalties.to_dict(),
        "metadata": {
            "episode_id": reduced.get("episode_id"),
            "n_variables": len(variables),
            "n_linear_terms": len(linear_terms),
            "n_quadratic_terms": len(quadratic_terms),
        },
    }


def qubo_energy(qubo: dict[str, Any], bitvec: list[int]) -> float:
    e = float(qubo.get("constant", 0.0))
    for t in qubo.get("linear_terms", []):
        i = int(t["i"])
        e += float(t["value"]) * int(bitvec[i])
    for t in qubo.get("quadratic_terms", []):
        i = int(t["i"])
        j = int(t["j"])
        e += float(t["value"]) * int(bitvec[i]) * int(bitvec[j])
    return float(e)


def decode_selected_candidates(qubo: dict[str, Any], bitvec: list[int]) -> list[str]:
    vars_ = qubo["variables"]
    return [vars_[i] for i, b in enumerate(bitvec) if int(b) == 1]
