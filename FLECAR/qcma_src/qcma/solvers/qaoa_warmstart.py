"""Cold-start and warm-start QAOA solvers on compiled QUBO.

Implementation note:
- Uses qiskit core + scipy only (no qiskit_optimization dependency).
- Keeps the same external solver interface used by the QCMA pipeline.
"""

from __future__ import annotations

import math
from time import perf_counter
from typing import Any

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from qcma.compiler.qubo_builder import decode_selected_candidates, qubo_energy


class _TimeoutError(RuntimeError):
    pass


def qubo_to_quadratic_program(qubo: dict[str, Any]) -> dict[str, Any]:
    """Compatibility shim retained for callers; returns a structured dict."""
    return {
        "n_variables": len(qubo["variables"]),
        "linear_terms": list(qubo.get("linear_terms", [])),
        "quadratic_terms": list(qubo.get("quadratic_terms", [])),
        "constant": float(qubo.get("constant", 0.0)),
    }


def _qubo_to_ising_coeffs(qubo: dict[str, Any]) -> tuple[np.ndarray, dict[tuple[int, int], float]]:
    n = len(qubo["variables"])
    z = np.zeros(n, dtype=float)
    zz: dict[tuple[int, int], float] = {}

    for t in qubo.get("linear_terms", []):
        i = int(t["i"])
        v = float(t["value"])
        z[i] += -0.5 * v

    for t in qubo.get("quadratic_terms", []):
        i = int(t["i"])
        j = int(t["j"])
        v = float(t["value"])
        z[i] += -0.25 * v
        z[j] += -0.25 * v
        key = (i, j) if i < j else (j, i)
        zz[key] = zz.get(key, 0.0) + 0.25 * v

    return z, zz


def _energy_table(qubo: dict[str, Any]) -> np.ndarray:
    n = len(qubo["variables"])
    size = 1 << n
    out = np.zeros(size, dtype=float)
    for s in range(size):
        bits = [(s >> i) & 1 for i in range(n)]
        out[s] = qubo_energy(qubo, bits)
    return out


def _seed_bits_from_selection(qubo: dict[str, Any], selected: set[str]) -> list[int]:
    vars_ = qubo["variables"]
    return [1 if v in selected else 0 for v in vars_]


def _build_qaoa_circuit(
    *,
    n: int,
    z_terms: np.ndarray,
    zz_terms: dict[tuple[int, int], float],
    gammas: np.ndarray,
    betas: np.ndarray,
    init_probs: np.ndarray | None,
) -> QuantumCircuit:
    qc = QuantumCircuit(n)

    # Initial state.
    if init_probs is None:
        for i in range(n):
            qc.h(i)
    else:
        for i in range(n):
            p1 = float(np.clip(init_probs[i], 1e-6, 1 - 1e-6))
            theta = 2.0 * math.asin(math.sqrt(p1))
            qc.ry(theta, i)

    p = len(gammas)
    for layer in range(p):
        g = float(gammas[layer])
        b = float(betas[layer])

        # Cost unitary exp(-i gamma H_C)
        for i in range(n):
            coeff = float(z_terms[i])
            if abs(coeff) > 1e-12:
                qc.rz(2.0 * g * coeff, i)
        for (i, j), coeff in zz_terms.items():
            if abs(coeff) > 1e-12:
                qc.rzz(2.0 * g * coeff, i, j)

        # X-mixer exp(-i beta sum X)
        for i in range(n):
            qc.rx(2.0 * b, i)

    return qc


def _best_bitstring_from_probs(probs: np.ndarray, energies: np.ndarray, top_k: int = 64) -> list[int]:
    n = int(round(math.log2(len(probs)))) if len(probs) > 1 else 1
    k = min(top_k, len(probs))
    # Candidate set = top-k most probable states.
    idx = np.argpartition(probs, -k)[-k:]
    best_state = min(idx, key=lambda s: (energies[s], -probs[s]))
    return [(int(best_state) >> i) & 1 for i in range(n)]


def _run_qaoa(
    qubo: dict[str, Any],
    *,
    name: str,
    reps: int,
    maxiter: int,
    seed: int,
    wall_clock_timeout_sec: float | None,
    init_probs: np.ndarray | None,
    optimizer_variant: str,
    init_strategy: str,
    estimator_mode: str,
    shots: int,
) -> dict[str, Any]:
    t0 = perf_counter()
    n = len(qubo["variables"])
    reps_eff = max(1, int(reps))
    maxiter_eff = max(1, int(maxiter))

    z_terms, zz_terms = _qubo_to_ising_coeffs(qubo)
    energies = _energy_table(qubo)

    rng = np.random.default_rng(seed)
    if init_strategy == "zeros":
        x0 = np.zeros(2 * reps_eff, dtype=float)
    elif init_strategy == "small_random":
        x0 = np.concatenate(
            [
                rng.uniform(0.0, 0.1, size=reps_eff),
                rng.uniform(0.0, 0.1, size=reps_eff),
            ]
        )
    else:
        x0 = np.concatenate(
            [
                rng.uniform(0.0, 0.3, size=reps_eff),
                rng.uniform(0.0, 0.3, size=reps_eff),
            ]
        )

    timeout_sec = None
    if wall_clock_timeout_sec is not None and wall_clock_timeout_sec > 0:
        timeout_sec = float(wall_clock_timeout_sec)

    eval_count = 0

    def objective(x: np.ndarray) -> float:
        nonlocal eval_count
        eval_count += 1
        if timeout_sec is not None and (perf_counter() - t0) > timeout_sec:
            raise _TimeoutError("QAOA objective timeout")

        gammas = np.array(x[:reps_eff], dtype=float)
        betas = np.array(x[reps_eff:], dtype=float)
        qc = _build_qaoa_circuit(
            n=n,
            z_terms=z_terms,
            zz_terms=zz_terms,
            gammas=gammas,
            betas=betas,
            init_probs=init_probs,
        )
        sv = Statevector.from_instruction(qc)
        probs = np.abs(sv.data) ** 2
        if estimator_mode == "shot" and int(shots) > 0:
            local_rng = np.random.default_rng(seed + eval_count)
            idx = local_rng.choice(len(probs), size=int(shots), p=probs)
            return float(np.mean(energies[idx]))
        return float(np.dot(probs, energies))

    try:
        x_best = np.array(x0, dtype=float)
        best_val = objective(x_best)

        if optimizer_variant == "random_restart":
            n_restarts = max(2, min(6, maxiter_eff // 2 if maxiter_eff >= 2 else 2))
            steps = max(1, maxiter_eff // n_restarts)
            for r in range(n_restarts):
                if timeout_sec is not None and (perf_counter() - t0) > timeout_sec:
                    raise _TimeoutError("QAOA optimization timeout")
                x_cur = np.concatenate(
                    [
                        rng.uniform(0.0, math.pi, size=reps_eff),
                        rng.uniform(0.0, math.pi / 2.0, size=reps_eff),
                    ]
                )
                val_cur = objective(x_cur)
                if val_cur < best_val:
                    best_val = val_cur
                    x_best = x_cur
                for it in range(steps):
                    sigma = max(0.03, 0.22 * (0.9**it))
                    x_try = x_cur + rng.normal(0.0, sigma, size=x_cur.shape)
                    x_try[:reps_eff] = np.clip(x_try[:reps_eff], 0.0, math.pi)
                    x_try[reps_eff:] = np.clip(x_try[reps_eff:], 0.0, math.pi / 2.0)
                    val = objective(x_try)
                    if val < val_cur:
                        val_cur = val
                        x_cur = x_try
                        if val_cur < best_val:
                            best_val = val_cur
                            x_best = x_cur
        else:
            # Budget-bounded randomized local search over QAOA parameters.
            for it in range(maxiter_eff):
                if timeout_sec is not None and (perf_counter() - t0) > timeout_sec:
                    raise _TimeoutError("QAOA optimization timeout")
                sigma = max(0.03, 0.25 * (0.92**it))
                x_try = x_best + rng.normal(0.0, sigma, size=x_best.shape)
                # gamma in [0, pi], beta in [0, pi/2]
                x_try[:reps_eff] = np.clip(x_try[:reps_eff], 0.0, math.pi)
                x_try[reps_eff:] = np.clip(x_try[reps_eff:], 0.0, math.pi / 2.0)
                val = objective(x_try)
                if val < best_val:
                    best_val = val
                    x_best = x_try

        gammas = x_best[:reps_eff]
        betas = x_best[reps_eff:]

        qc_best = _build_qaoa_circuit(
            n=n,
            z_terms=z_terms,
            zz_terms=zz_terms,
            gammas=gammas,
            betas=betas,
            init_probs=init_probs,
        )
        sv_best = Statevector.from_instruction(qc_best)
        probs_best = np.abs(sv_best.data) ** 2
        bits = _best_bitstring_from_probs(probs_best, energies)

        selected = decode_selected_candidates(qubo, bits)
        return {
            "solver": name,
            "selected_candidate_ids": selected,
            "runtime_sec": perf_counter() - t0,
            "objective": float(qubo_energy(qubo, bits)),
            "solver_status": "OPTIMIZED",
            "solver_meta": {
                "backend": "statevector_custom_qaoa",
                "reps": reps_eff,
                "maxiter": maxiter_eff,
                "n_variables": n,
                "eval_count": int(eval_count),
                "fun_expectation": float(best_val),
                "optimizer_variant": optimizer_variant,
                "init_strategy": init_strategy,
                "estimator_mode": estimator_mode,
                "shots": int(shots),
            },
        }
    except _TimeoutError:
        return {
            "solver": name,
            "selected_candidate_ids": [],
            "runtime_sec": perf_counter() - t0,
            "objective": float("inf"),
            "solver_status": "TIMEOUT",
            "solver_meta": {
                "backend": "statevector_custom_qaoa",
                "reps": reps_eff,
                "maxiter": maxiter_eff,
                "n_variables": n,
                "eval_count": int(eval_count),
                "timeout_sec": timeout_sec,
                "optimizer_variant": optimizer_variant,
                "init_strategy": init_strategy,
                "estimator_mode": estimator_mode,
                "shots": int(shots),
            },
        }
    except Exception as err:
        return {
            "solver": name,
            "selected_candidate_ids": [],
            "runtime_sec": perf_counter() - t0,
            "objective": float("inf"),
            "solver_status": "ERROR",
            "solver_meta": {
                "backend": "statevector_custom_qaoa",
                "reps": reps_eff,
                "maxiter": maxiter_eff,
                "n_variables": n,
                "eval_count": int(eval_count),
                "error": f"{type(err).__name__}: {err}",
                "optimizer_variant": optimizer_variant,
                "init_strategy": init_strategy,
                "estimator_mode": estimator_mode,
                "shots": int(shots),
            },
        }


def solve_qaoa_cold_start(
    qubo: dict[str, Any],
    *,
    reps: int = 1,
    maxiter: int = 50,
    seed: int = 7,
    wall_clock_timeout_sec: float | None = None,
    optimizer_variant: str = "random_local",
    init_strategy: str = "random",
    estimator_mode: str = "exact",
    shots: int = 0,
) -> dict[str, Any]:
    return _run_qaoa(
        qubo,
        name="qaoa_cold",
        reps=reps,
        maxiter=maxiter,
        seed=seed,
        wall_clock_timeout_sec=wall_clock_timeout_sec,
        init_probs=None,
        optimizer_variant=optimizer_variant,
        init_strategy=init_strategy,
        estimator_mode=estimator_mode,
        shots=shots,
    )


def solve_qaoa_warm_start(
    qubo: dict[str, Any],
    *,
    reps: int = 1,
    maxiter: int = 50,
    seed: int = 7,
    epsilon: float = 0.25,
    pre_solver_iter: int = 30,
    wall_clock_timeout_sec: float | None = None,
    optimizer_variant: str = "random_local",
    init_strategy: str = "random",
    estimator_mode: str = "exact",
    shots: int = 0,
) -> dict[str, Any]:
    # Warm-start proxy from seed-biased initial state.
    # If QUBO was compiled with prev_selection + lambda_repair, this state aligns
    # with the same guidance direction while preserving shared evaluator fairness.
    vars_ = qubo["variables"]
    n = len(vars_)

    # Use explicit seed from QUBO repair metadata when available; fallback to zeros.
    # For now, infer from linear bias signs when repair term is present.
    # Positive linear bias -> prefer x=0; negative -> prefer x=1.
    bits_guess = [0 for _ in range(n)]
    lin = {int(t["i"]): float(t["value"]) for t in qubo.get("linear_terms", [])}
    for i in range(n):
        bits_guess[i] = 1 if lin.get(i, 0.0) < 0 else 0

    eps = float(np.clip(epsilon, 1e-3, 0.49))
    init_probs = np.array([1.0 - eps if b == 1 else eps for b in bits_guess], dtype=float)

    out = _run_qaoa(
        qubo,
        name="qaoa_warm",
        reps=reps,
        maxiter=maxiter,
        seed=seed,
        wall_clock_timeout_sec=wall_clock_timeout_sec,
        init_probs=init_probs,
        optimizer_variant=optimizer_variant,
        init_strategy=init_strategy,
        estimator_mode=estimator_mode,
        shots=shots,
    )
    out.setdefault("solver_meta", {})["epsilon"] = float(epsilon)
    out.setdefault("solver_meta", {})["pre_solver_iter"] = int(pre_solver_iter)
    return out
