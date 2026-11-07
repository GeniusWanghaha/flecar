from __future__ import annotations

from collections import defaultdict
from typing import Any

from qcma.solvers.common import is_feasible

from flecar.dispatch import check_dispatchability, envelope_validity_check
from flecar.repair.normalization import (
    normalize_fraction,
    normalize_margin_deficit,
    normalize_nonnegative_surrogate,
)


def _candidate_to_subtask(reduced: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for sid, cids in reduced.get("unique_groups", {}).items():
        for cid in cids:
            out[str(cid)] = str(sid)
    return out


def _subtask_has_dispatchable_backup(
    reduced: dict[str, Any],
    envelope: dict[str, Any],
    sid: str,
    banned: set[str],
) -> bool:
    windows = envelope.get("windows", {})
    if sid not in windows:
        return False
    w = windows[sid]
    l = float(w.get("l", 0.0))
    u = float(w.get("u", 0.0))
    nodes = reduced.get("nodes", {})
    for cid in reduced.get("unique_groups", {}).get(sid, []):
        cc = str(cid)
        if cc in banned:
            continue
        node = nodes.get(cc)
        if node is None:
            continue
        s = float(node.get("start", 0.0))
        if l - 1e-6 <= s <= u + 1e-6:
            return True
    return False


def event_diagnostics(
    reduced: dict[str, Any],
    envelope: dict[str, Any],
    selected_ids: list[str],
    *,
    event_type: str,
    trigger_time: int,
    banned_candidates: set[str],
) -> dict[str, Any]:
    banned = {str(x) for x in banned_candidates}
    impacted = [str(cid) for cid in selected_ids if str(cid) in banned]
    selected_after_filter = [str(cid) for cid in selected_ids if str(cid) not in banned]
    feasible_after = is_feasible(reduced, selected_after_filter)

    c2s = _candidate_to_subtask(reduced)
    impacted_subtasks = sorted({c2s.get(str(cid), "") for cid in impacted if c2s.get(str(cid), "")})

    validity = envelope_validity_check(envelope)
    disp = check_dispatchability(envelope)
    ctrl_margin = float(disp.get("controllability_margin", 0.0))

    local_slacks = []
    windows = envelope.get("windows", {})
    for sid in impacted_subtasks:
        w = windows.get(sid)
        if w:
            local_slacks.append(max(0.0, float(w.get("u", 0.0)) - float(w.get("l", 0.0))))

    # Subtask-level absorbability under envelope windows.
    backup_failures = 0
    for sid in impacted_subtasks:
        if not _subtask_has_dispatchable_backup(reduced, envelope, sid, banned):
            backup_failures += 1

    sync_pairs = envelope.get("constraints", {}).get("sync_pairs", [])
    imp_set = set(impacted_subtasks)
    synchronized_risk = 0.0
    for pair in sync_pairs:
        if not isinstance(pair, list) or len(pair) != 2:
            continue
        if str(pair[0]) in imp_set or str(pair[1]) in imp_set:
            synchronized_risk += 1.0

    # Resource criticality shift: impacted resource share vs previous selected resources.
    nodes = reduced.get("nodes", {})
    prev_resources = []
    impacted_resources = []
    for cid in selected_ids:
        prev_resources.extend(str(r) for r in nodes.get(str(cid), {}).get("resources", []))
    for cid in impacted:
        impacted_resources.extend(str(r) for r in nodes.get(str(cid), {}).get("resources", []))
    prev_freq: dict[str, int] = defaultdict(int)
    for r in prev_resources:
        prev_freq[r] += 1
    impacted_freq: dict[str, int] = defaultdict(int)
    for r in impacted_resources:
        impacted_freq[r] += 1

    shift = 0.0
    for r, c in impacted_freq.items():
        shift += float(c) / float(max(1, prev_freq.get(r, 1)))
    resource_criticality_shift = shift / float(max(1, len(impacted_freq))) if impacted_freq else 0.0

    base_event_risk = {
        "task_delay": 0.45,
        "task_failure": 0.95,
        "resource_occupation_change": 0.80,
        "object_state_change": 0.55,
        "temporary_agent_unavailability": 0.90,
        "agent_unavailable": 0.90,
    }.get(str(event_type), 0.60)

    local_slack_consumption = 0.0
    if local_slacks:
        avg_local_slack = sum(local_slacks) / float(max(1, len(local_slacks)))
        local_slack_consumption = min(1.0, float(len(impacted_subtasks)) / float(max(1.0, avg_local_slack + 1.0)))

    future_conflict_forecast_raw = (
        float(base_event_risk)
        + 0.20 * float(len(impacted_subtasks))
        + 0.25 * float(backup_failures)
        + 0.15 * float(synchronized_risk)
        + 0.15 * float(resource_criticality_shift)
    )

    envelope_violation_severity_raw = (
        (0.0 if validity.get("valid", False) else 1.0)
        + (0.0 if disp.get("dispatchable", False) else 1.0)
        + max(0.0, -ctrl_margin)
        + 0.35 * float(backup_failures)
        + 0.15 * float(len(impacted_subtasks))
    )

    diagnostic_v = normalize_nonnegative_surrogate(envelope_violation_severity_raw, scale=1.0)
    diagnostic_m = normalize_margin_deficit(ctrl_margin, scale=1.0)
    diagnostic_s = normalize_fraction(
        len(impacted_subtasks),
        (sum(local_slacks) + 1.0) if local_slacks else 1.0,
        default=0.0,
    )
    diagnostic_f = normalize_nonnegative_surrogate(future_conflict_forecast_raw, scale=1.0)
    diagnostic_y = normalize_nonnegative_surrogate(synchronized_risk, scale=1.0)
    diagnostic_r = normalize_nonnegative_surrogate(resource_criticality_shift, scale=1.0)

    impacted_n = max(1, len(impacted_subtasks))
    backup_ratio = float(backup_failures) / float(impacted_n)
    dispatchable_after_event = bool(
        validity.get("valid", False)
        and disp.get("dispatchable", False)
        and feasible_after
        and (
            backup_failures == 0
            or (backup_ratio <= 0.34 and float(ctrl_margin) >= -0.15)
        )
    )

    return {
        "event_type": str(event_type),
        "trigger_time": int(trigger_time),
        "impacted_candidates": sorted(impacted),
        "impacted_subtasks": impacted_subtasks,
        "diagnostic_vector": [
            float(diagnostic_v),
            float(diagnostic_m),
            float(diagnostic_s),
            float(diagnostic_f),
            float(diagnostic_y),
            float(diagnostic_r),
        ],
        "envelope_violation_severity": float(diagnostic_v),
        "envelope_violation_severity_raw": float(envelope_violation_severity_raw),
        "controllability_margin": float(ctrl_margin),
        "controllability_margin_severity": float(diagnostic_m),
        "local_slack_consumption": float(diagnostic_s),
        "local_slack_consumption_raw": float(local_slack_consumption),
        "future_conflict_forecast": float(diagnostic_f),
        "future_conflict_forecast_raw": float(future_conflict_forecast_raw),
        "synchronized_task_risk": float(diagnostic_y),
        "synchronized_task_risk_raw": float(synchronized_risk),
        "resource_criticality_shift": float(diagnostic_r),
        "resource_criticality_shift_raw": float(resource_criticality_shift),
        "dispatchable_after_event": bool(dispatchable_after_event),
        "backup_failures": int(backup_failures),
        "backup_failure_ratio": float(backup_ratio),
        "feasible_after_ban_filter": bool(feasible_after),
    }
