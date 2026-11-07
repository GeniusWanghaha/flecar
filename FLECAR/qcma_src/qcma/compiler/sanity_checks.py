"""Small-scale QUBO sanity checks via brute-force enumeration."""

from __future__ import annotations

import itertools
from typing import Any

from qcma.compiler.qubo_builder import compile_qubo, decode_selected_candidates, qubo_energy
from qcma.solvers.common import is_feasible, objective_from_selection


def brute_force_best(qubo: dict[str, Any]) -> tuple[list[int], float]:
    n = len(qubo["variables"])
    best_bits: list[int] | None = None
    best_e = float("inf")
    for bits in itertools.product([0, 1], repeat=n):
        b = list(bits)
        e = qubo_energy(qubo, b)
        if e < best_e:
            best_e = e
            best_bits = b
    assert best_bits is not None
    return best_bits, best_e


def sanity_check_qubo_feasibility_alignment(reduced: dict[str, Any]) -> dict[str, Any]:
    """For small graphs, verify min-energy solution is feasible under high penalties."""
    qubo = compile_qubo(reduced)
    n = len(qubo["variables"])
    if n > 22:
        raise ValueError(f"Brute-force sanity check only for <=22 vars, got {n}")
    bits, e = brute_force_best(qubo)
    sel = decode_selected_candidates(qubo, bits)
    best_energy_solution_feasible = is_feasible(reduced, sel)
    obj = objective_from_selection(reduced, sel) if sel else float("inf")

    has_feasible = False
    for cand_bits in itertools.product([0, 1], repeat=n):
        cand_sel = decode_selected_candidates(qubo, list(cand_bits))
        if is_feasible(reduced, cand_sel):
            has_feasible = True
            break

    return {
        "n_variables": n,
        "energy": e,
        "selected": sel,
        "best_energy_solution_feasible": best_energy_solution_feasible,
        "has_feasible_solution": has_feasible,
        "sanity_pass": (not has_feasible) or best_energy_solution_feasible,
        "objective": obj,
    }
