"""Event-driven repair simulator for offline rolling-horizon studies."""

from __future__ import annotations

import random
from collections import defaultdict, deque
from typing import Any


EVENT_TYPES = [
    "task_delay",
    "task_failure",
    "resource_occupation_change",
    "object_state_change",
    "agent_unavailable",
]


def selected_by_subtask(reduced: dict[str, Any], selected: list[str]) -> dict[str, str]:
    out = {}
    for cid in selected:
        sid = reduced["nodes"][cid]["subtask_id"]
        out[sid] = cid
    return out


def frozen_subtasks_at_time(
    reduced: dict[str, Any], selected: list[str], trigger_time: int
) -> set[str]:
    frozen = set()
    for cid in selected:
        node = reduced["nodes"][cid]
        if int(node["start"]) < int(trigger_time):
            frozen.add(node["subtask_id"])
    return frozen


def edit_distance_by_subtask(
    reduced: dict[str, Any], prev_selected: list[str], new_selected: list[str]
) -> int:
    p = selected_by_subtask(reduced, prev_selected)
    n = selected_by_subtask(reduced, new_selected)
    subtasks = reduced["unique_groups"].keys()
    edits = 0
    for sid in subtasks:
        if p.get(sid) != n.get(sid):
            edits += 1
    return edits


def sample_trigger_times(reduced: dict[str, Any], selected: list[str], n_events: int) -> list[int]:
    starts = sorted({int(reduced["nodes"][cid]["start"]) for cid in selected})
    if not starts:
        return [0] * n_events
    if len(starts) >= n_events:
        return starts[:n_events]
    out = starts[:]
    while len(out) < n_events:
        out.append(starts[-1])
    return out


def apply_event_to_bans(
    reduced: dict[str, Any],
    prev_selected: list[str],
    *,
    trigger_time: int,
    event_type: str,
    rng: random.Random,
) -> set[str]:
    nodes = reduced["nodes"]
    by_subtask = selected_by_subtask(reduced, prev_selected)
    all_cids = list(nodes.keys())
    if not all_cids:
        return set()

    banned: set[str] = set()
    # Prefer currently selected candidates so events are more likely to affect
    # the active plan and create a meaningful repair-vs-replan contrast.
    chosen_pool = prev_selected if prev_selected else all_cids
    chosen_cid = chosen_pool[rng.randrange(len(chosen_pool))]
    chosen_sid = nodes[chosen_cid]["subtask_id"]

    if event_type == "task_delay":
        # Ban early-start candidates of affected subtask to force postponement/repair edits.
        for cid in reduced["unique_groups"][chosen_sid]:
            if int(nodes[cid]["start"]) <= int(trigger_time):
                banned.add(cid)
    elif event_type == "task_failure":
        # Ban previously selected candidate for this subtask.
        if chosen_sid in by_subtask:
            banned.add(by_subtask[chosen_sid])
    elif event_type == "resource_occupation_change":
        target_resource = None
        for cid in reduced["unique_groups"][chosen_sid]:
            resources = nodes[cid].get("resources", [])
            if resources:
                target_resource = resources[0]
                break
        if target_resource is not None:
            for cid, node in nodes.items():
                if target_resource in node.get("resources", []):
                    banned.add(cid)
    elif event_type == "object_state_change":
        # Ban half candidates in one subtask.
        cands = list(reduced["unique_groups"][chosen_sid])
        rng.shuffle(cands)
        for cid in cands[: max(1, len(cands) // 2)]:
            banned.add(cid)
    elif event_type == "agent_unavailable":
        # Ban candidates from one agent at/after trigger time.
        agent = nodes[chosen_cid]["agent_id"]
        for cid, node in nodes.items():
            if node["agent_id"] == agent and int(node["start"]) >= int(trigger_time):
                banned.add(cid)
    return banned


def event_adaptive_edit_penalty(base_penalty: float, event_type: str) -> float:
    """Event-conditioned edit regularization.

    Larger penalties preserve schedule stability when disturbance is localized
    (e.g., object-state changes), while lower penalties allow broader
    adaptation for agent/resource outages.
    """
    et = str(event_type)
    mult = {
        "task_delay": 0.9,
        "task_failure": 1.35,
        "resource_occupation_change": 1.1,
        "object_state_change": 1.2,
        "agent_unavailable": 0.8,
    }.get(et, 1.0)
    return float(base_penalty) * float(mult)


def _build_subtask_adjacency(reduced: dict[str, Any]) -> dict[str, set[str]]:
    adj: dict[str, set[str]] = defaultdict(set)
    subtasks = set(reduced.get("unique_groups", {}).keys())
    for sid in subtasks:
        adj.setdefault(str(sid), set())

    # Precedence neighborhood.
    for pair in reduced.get("precedence_pairs", []):
        if not isinstance(pair, list) or len(pair) != 2:
            continue
        u = str(pair[0])
        v = str(pair[1])
        if u not in adj or v not in adj:
            continue
        adj[u].add(v)
        adj[v].add(u)

    # Conflict neighborhood projected from candidate edges.
    nodes = reduced.get("nodes", {})
    for key in ("conflict_edges", "precedence_invalid_edges", "sync_invalid_edges"):
        for u, v in reduced.get(key, []):
            su = str(nodes.get(u, {}).get("subtask_id", ""))
            sv = str(nodes.get(v, {}).get("subtask_id", ""))
            if not su or not sv or su == sv:
                continue
            if su in adj and sv in adj:
                adj[su].add(sv)
                adj[sv].add(su)
    return adj


def adaptive_repair_window_subtasks(
    reduced: dict[str, Any],
    *,
    banned_candidates: set[str],
    event_type: str,
    frozen_subtasks: set[str],
    default_radius: int = 1,
) -> set[str]:
    """Event-adaptive editable subtask window.

    Builds a subtask neighborhood from precedence/conflict adjacency and expands
    from event-affected subtasks with event-dependent radii.
    """
    nodes = reduced.get("nodes", {})
    groups = reduced.get("unique_groups", {})
    all_subtasks = set(groups.keys())
    if not all_subtasks:
        return set()

    affected = {
        str(nodes[cid]["subtask_id"])
        for cid in banned_candidates
        if cid in nodes and str(nodes[cid].get("subtask_id", "")) in all_subtasks
    }
    if not affected:
        # Fallback to all editable subtasks if bans are empty/unmapped.
        return set(all_subtasks - set(frozen_subtasks))

    radius_by_event = {
        "task_delay": max(1, default_radius),
        "task_failure": max(1, default_radius + 1),
        "resource_occupation_change": max(1, default_radius + 1),
        "object_state_change": max(1, default_radius),
        "agent_unavailable": max(1, default_radius + 1),
    }
    radius = int(radius_by_event.get(str(event_type), max(1, default_radius)))
    adj = _build_subtask_adjacency(reduced)

    editable: set[str] = set()
    q: deque[tuple[str, int]] = deque((sid, 0) for sid in affected)
    seen = set(affected)
    while q:
        sid, d = q.popleft()
        if sid not in frozen_subtasks:
            editable.add(sid)
        if d >= radius:
            continue
        for nxt in adj.get(sid, set()):
            if nxt in seen:
                continue
            seen.add(nxt)
            q.append((nxt, d + 1))

    if not editable:
        editable = set(all_subtasks - set(frozen_subtasks))
    return editable


def adaptive_scope_controller(
    reduced: dict[str, Any],
    prev_selected: list[str],
    *,
    trigger_time: int,
    banned_candidates: set[str],
    event_type: str,
    base_edit_penalty: float,
    default_radius: int = 1,
) -> dict[str, Any]:
    """Module B1: event-adaptive repair scope + penalty controller.

    Returns controlled frozen/editable sets and adaptive penalty while keeping
    the same shared repair objective/evaluator contract.
    """
    nodes = reduced.get("nodes", {})
    all_subtasks = set(reduced.get("unique_groups", {}).keys())
    frozen = frozen_subtasks_at_time(reduced, prev_selected, int(trigger_time))
    if not all_subtasks:
        return {
            "frozen_subtasks": frozen,
            "editable_subtasks": set(),
            "edit_penalty": float(base_edit_penalty),
            "severity": 0.0,
            "radius": int(default_radius),
            "unfroze": 0,
        }

    severity = float(len(banned_candidates)) / float(max(1, len(nodes)))
    heavy_events = {"agent_unavailable", "resource_occupation_change", "task_failure"}
    radius = int(default_radius)
    if str(event_type) in heavy_events:
        radius += 1
    if severity >= 0.18:
        radius += 1
    radius = max(1, radius)

    editable = adaptive_repair_window_subtasks(
        reduced,
        banned_candidates=banned_candidates,
        event_type=event_type,
        frozen_subtasks=frozen,
        default_radius=radius,
    )

    # Prefix-freezing granularity control:
    # in severe outages, allow limited unfreezing near affected frontier.
    frozen_adj = set(frozen)
    unfroze = 0
    if severity >= 0.20 or str(event_type) in {"agent_unavailable", "resource_occupation_change"}:
        adj = _build_subtask_adjacency(reduced)
        affected = {
            str(nodes[cid]["subtask_id"])
            for cid in banned_candidates
            if cid in nodes and str(nodes[cid].get("subtask_id", "")) in all_subtasks
        }
        frontier = set()
        for sid in affected:
            frontier.update(adj.get(sid, set()))
        # only relax a bounded portion of frozen prefix.
        relax_budget = max(1, int(round(0.2 * max(1, len(frozen_adj)))))
        relaxable = [sid for sid in sorted(frontier) if sid in frozen_adj]
        for sid in relaxable[:relax_budget]:
            frozen_adj.remove(sid)
            editable.add(sid)
            unfroze += 1

    pen = event_adaptive_edit_penalty(float(base_edit_penalty), str(event_type))
    if severity >= 0.20:
        pen *= 0.85  # severe events favor broader adaptation
    elif severity <= 0.05 and str(event_type) == "object_state_change":
        pen *= 1.10  # localized events favor stability

    return {
        "frozen_subtasks": frozen_adj,
        "editable_subtasks": editable,
        "edit_penalty": float(pen),
        "severity": float(severity),
        "radius": int(radius),
        "unfroze": int(unfroze),
    }
