from __future__ import annotations

from typing import Any

REQUIRED_INSTANCE_KEYS = {
    "episode_id",
    "task_text",
    "agents",
    "subtasks",
    "precedence",
    "candidate_assignments",
}

REQUIRED_XI_KEYS = {
    "duration_interval_uncertainty",
    "resource_occupancy_fluctuation",
    "temporary_agent_unavailability",
    "task_failure_hazard_prior",
    "object_state_disturbance_prior",
}


class DataContractError(ValueError):
    pass


def _extract_xi(instance: dict[str, Any]) -> dict[str, Any]:
    if isinstance(instance.get("Xi"), dict):
        return dict(instance["Xi"])
    meta = instance.get("metadata", {})
    if isinstance(meta, dict) and isinstance(meta.get("Xi"), dict):
        return dict(meta["Xi"])
    return {}


def validate_instance_schema(instance: dict[str, Any]) -> None:
    missing = sorted(k for k in REQUIRED_INSTANCE_KEYS if k not in instance)
    if missing:
        raise DataContractError(f"missing instance keys: {missing}")

    if not isinstance(instance["candidate_assignments"], list) or not instance["candidate_assignments"]:
        raise DataContractError("candidate_assignments must be non-empty list")

    cand_ids: set[str] = set()
    subtask_ids: set[str] = {str(s["id"]) for s in instance["subtasks"]}
    for c in instance["candidate_assignments"]:
        cid = str(c.get("candidate_id", ""))
        sid = str(c.get("subtask_id", ""))
        if not cid or cid in cand_ids:
            raise DataContractError(f"invalid or duplicate candidate_id: {cid}")
        cand_ids.add(cid)
        if sid not in subtask_ids:
            raise DataContractError(f"candidate references unknown subtask: {sid}")
        for key in ("agent_id", "start", "duration", "cost"):
            if key not in c:
                raise DataContractError(f"candidate missing {key}: {cid}")

    xi = _extract_xi(instance)
    xi_missing = sorted(k for k in REQUIRED_XI_KEYS if k not in xi)
    if xi_missing:
        raise DataContractError(f"missing Xi keys: {xi_missing}")


def validate_dataset(instances: list[dict[str, Any]]) -> None:
    episode_ids: set[str] = set()
    for inst in instances:
        validate_instance_schema(inst)
        eid = str(inst["episode_id"])
        if eid in episode_ids:
            raise DataContractError(f"duplicate episode_id: {eid}")
        episode_ids.add(eid)
