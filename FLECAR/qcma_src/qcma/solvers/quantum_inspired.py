"""Quantum-inspired baseline: simulated annealing on QUBO."""

from __future__ import annotations

import math
import random
from time import perf_counter
from typing import Any

from qcma.compiler.qubo_builder import decode_selected_candidates, qubo_energy


def solve_qubo_simulated_annealing(
    qubo: dict[str, Any],
    *,
    time_limit_sec: float = 1.0,
    steps: int = 5000,
    temp_start: float = 5.0,
    temp_end: float = 0.01,
    seed: int = 0,
    init_bits: list[int] | None = None,
    preserve_onehot: bool = True,
) -> dict[str, Any]:
    rng = random.Random(seed)
    t0 = perf_counter()
    n = len(qubo["variables"])
    groups = list(qubo.get("group_indices", {}).values())

    if init_bits is not None:
        bits = list(init_bits)
    elif preserve_onehot and groups:
        bits = [0 for _ in range(n)]
        for g in groups:
            bits[g[rng.randrange(len(g))]] = 1
    else:
        bits = [rng.randint(0, 1) for _ in range(n)]
    e = qubo_energy(qubo, bits)
    best_bits = list(bits)
    best_e = e

    steps_done = 0
    for k in range(max(1, steps)):
        if perf_counter() - t0 > time_limit_sec:
            break

        alpha = k / max(1, steps - 1)
        temp = temp_start * (1 - alpha) + temp_end * alpha

        changed: list[int] = []
        if preserve_onehot and groups:
            g = groups[rng.randrange(len(groups))]
            current = [idx for idx in g if bits[idx] == 1]
            old_idx = current[0] if current else None
            new_idx = g[rng.randrange(len(g))]
            if old_idx is not None and old_idx == new_idx:
                steps_done += 1
                continue
            if old_idx is not None:
                bits[old_idx] = 0
                changed.append(old_idx)
            bits[new_idx] = 1
            changed.append(new_idx)
        else:
            i = rng.randrange(n)
            bits[i] = 1 - bits[i]
            changed.append(i)

        e_new = qubo_energy(qubo, bits)
        delta = e_new - e
        accept = delta <= 0 or rng.random() < math.exp(-delta / max(temp, 1e-9))
        if accept:
            e = e_new
            if e < best_e:
                best_e = e
                best_bits = list(bits)
        else:
            for idx in changed:
                bits[idx] = 1 - bits[idx]
        steps_done += 1

    runtime = perf_counter() - t0
    return {
        "solver": "quantum_inspired_sa",
        "selected_candidate_ids": decode_selected_candidates(qubo, best_bits),
        "qubo_energy": float(best_e),
        "runtime_sec": runtime,
        "steps_done": steps_done,
        "objective": float(best_e),
    }
