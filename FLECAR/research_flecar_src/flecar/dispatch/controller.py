from __future__ import annotations

from typing import Any

from qcma.solvers.common import is_feasible


def envelope_validity_check(envelope: dict[str, Any]) -> dict[str, Any]:
    checks = envelope.get("checks", {})
    validity = checks.get("validity", {}) if isinstance(checks, dict) else {}
    if validity:
        return {
            "valid": bool(validity.get("valid", False)),
            "reasons": list(validity.get("reasons", [])),
        }

    reasons: list[str] = []
    windows = envelope.get("windows", {})
    for sid, w in windows.items():
        if float(w.get("u", 0.0)) < float(w.get("l", 0.0)):
            reasons.append(f"invalid_window:{sid}")
    if not envelope.get("assignment"):
        reasons.append("empty_assignment")
    return {"valid": len(reasons) == 0, "reasons": reasons}


def check_dispatchability(envelope: dict[str, Any]) -> dict[str, Any]:
    checks = envelope.get("checks", {})
    dispatch = checks.get("dispatchability", {}) if isinstance(checks, dict) else {}
    metrics = envelope.get("metrics", {})

    if dispatch:
        dispatchable = bool(dispatch.get("dispatchable", False))
        ctrl = float(dispatch.get("controllability_margin", metrics.get("controllability_margin", 0.0)))
        csm = float(dispatch.get("critical_slack_margin", metrics.get("critical_slack_margin", 0.0)))
    else:
        dispatchable = bool(metrics.get("dispatchability_success", False))
        ctrl = float(metrics.get("controllability_margin", 0.0))
        csm = float(metrics.get("critical_slack_margin", 0.0))

    return {
        "dispatchable": bool(dispatchable and ctrl >= -1e-6),
        "controllability_margin": float(ctrl),
        "critical_slack_margin": float(csm),
    }


def _advance_windows(envelope: dict[str, Any], current_time: int) -> dict[str, Any]:
    windows = envelope.get("windows", {})
    if not windows:
        return envelope
    new_env = dict(envelope)
    new_windows = {}
    for sid, w in windows.items():
        l = float(w.get("l", 0.0))
        u = float(w.get("u", 0.0))
        dur = float(w.get("dur", 0.0))
        if u < float(current_time):
            new_windows[sid] = {"l": l, "u": u, "dur": dur, "closed": True}
        else:
            new_windows[sid] = {"l": max(l, float(current_time)), "u": u, "dur": dur, "closed": False}
    new_env["windows"] = new_windows
    if "W" in new_env:
        new_env["W"] = new_windows
    return new_env


def dispatch_step(
    reduced: dict[str, Any],
    envelope: dict[str, Any],
    selected_ids: list[str],
    *,
    banned_candidates: set[str] | None = None,
    current_time: int | None = None,
) -> dict[str, Any]:
    banned = {str(x) for x in (banned_candidates or set())}
    selection = [str(c) for c in selected_ids if str(c) not in banned]

    effective_env = envelope
    if current_time is not None:
        effective_env = _advance_windows(envelope, int(current_time))

    valid = envelope_validity_check(effective_env)
    disp = check_dispatchability(effective_env)
    feasible = is_feasible(reduced, selection)

    # Local execution-time dispatch interface: subtasks whose windows are currently open.
    dispatchable_now: list[str] = []
    t = float(current_time) if current_time is not None else None
    for sid, w in effective_env.get("windows", {}).items():
        l = float(w.get("l", 0.0))
        u = float(w.get("u", 0.0))
        if t is None or (l <= t <= u):
            dispatchable_now.append(str(sid))

    return {
        "dispatchable": bool(valid["valid"] and disp["dispatchable"] and feasible),
        "selection": selection,
        "feasible": bool(feasible),
        "envelope_valid": bool(valid["valid"]),
        "validity_reasons": list(valid["reasons"]),
        "controllability_margin": float(disp["controllability_margin"]),
        "dispatchable_subtasks_now": sorted(dispatchable_now),
        "effective_envelope": effective_env,
    }


def absorbable_disturbance_radius(envelope: dict[str, Any]) -> float:
    metrics = envelope.get("metrics", {})
    flex = float(metrics.get("flexibility_volume", 0.0))
    ctrl = max(0.0, float(metrics.get("controllability_margin", 0.0)))
    sync_tol = max(0.0, float(metrics.get("sync_tolerance", 0.0)))
    csm = max(0.0, float(metrics.get("critical_slack_margin", 0.0)))
    return float(0.28 * flex + 1.2 * ctrl + 0.4 * sync_tol + 0.4 * csm)
