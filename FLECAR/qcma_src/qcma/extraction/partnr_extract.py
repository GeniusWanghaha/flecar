"""PARTNR episode -> QCMA unified instance extraction."""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qcma.utils.io import dump_json, load_json_auto


@dataclass(frozen=True)
class AgentSpec:
    agent_id: str
    capabilities: set[str]


DEFAULT_AGENTS: list[AgentSpec] = [
    AgentSpec(
        agent_id="robot_0",
        capabilities={"navigate", "pick", "place", "open", "close", "inspect", "carry"},
    ),
    AgentSpec(
        agent_id="human_0",
        capabilities={"navigate", "carry", "open", "close", "inspect"},
    ),
]


def _capability_from_function_name(name: str) -> set[str]:
    n = name.lower()
    if "pick" in n or "grasp" in n:
        return {"pick"}
    if "place" in n or "put" in n:
        return {"place"}
    if "open" in n:
        return {"open"}
    if "close" in n:
        return {"close"}
    if "navigate" in n or "goto" in n or "move" in n:
        return {"navigate"}
    if "clean" in n or "wash" in n:
        return {"carry"}
    if "inspect" in n or "find" in n:
        return {"inspect"}
    return {"carry"}


def _duration_hint(function_name: str) -> int:
    n = function_name.lower()
    if "navigate" in n:
        return 4
    if "pick" in n or "place" in n:
        return 3
    if "open" in n or "close" in n:
        return 2
    return 3


def _extract_resource_tokens(args: dict[str, Any]) -> list[str]:
    tokens: list[str] = []
    for key, value in args.items():
        kl = key.lower()
        if "object" in kl or "receptacle" in kl or "furniture" in kl or "entity" in kl:
            if isinstance(value, str):
                tokens.append(value)
            elif isinstance(value, list):
                tokens.extend(str(v) for v in value)
    return sorted(set(tokens))


def _extract_zone(args: dict[str, Any]) -> str:
    for key, value in args.items():
        kl = key.lower()
        if "room" in kl or "zone" in kl or "region" in kl:
            if isinstance(value, str) and value.strip():
                return value
            if isinstance(value, list) and value:
                return str(value[0])
    return "global"


def _flatten_episodes(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, dict) and "episodes" in raw and isinstance(raw["episodes"], list):
        return [e for e in raw["episodes"] if isinstance(e, dict)]
    if isinstance(raw, list):
        return [e for e in raw if isinstance(e, dict)]
    raise ValueError("Unsupported dataset format. Expected dict with key 'episodes' or list.")


def _build_subtasks_from_episode(episode: dict[str, Any]) -> list[dict[str, Any]]:
    props = episode.get("evaluation_propositions", [])
    subtasks: list[dict[str, Any]] = []

    if isinstance(props, list) and props:
        for i, prop in enumerate(props):
            fn = str(prop.get("function_name", f"task_{i}"))
            req = sorted(_capability_from_function_name(fn))
            subtasks.append(
                {
                    "id": f"s{i}",
                    "label": fn,
                    "required_capabilities": req,
                    "duration_hint": _duration_hint(fn),
                    "metadata": {"args": prop.get("args", {})},
                }
            )
        return subtasks

    instruction = str(episode.get("instruction", "unknown task")).strip()
    chunks = [c.strip() for c in instruction.replace(",", " and ").split(" and ") if c.strip()]
    if not chunks:
        chunks = [instruction]
    for i, c in enumerate(chunks):
        subtasks.append(
            {
                "id": f"s{i}",
                "label": c,
                "required_capabilities": ["carry"],
                "duration_hint": 3,
                "metadata": {"args": {}},
            }
        )
    return subtasks


def _build_precedence(episode: dict[str, Any], n_subtasks: int) -> list[list[str]]:
    deps = episode.get("evaluation_proposition_dependencies", [])
    precedence: list[list[str]] = []

    if isinstance(deps, list):
        for dep in deps:
            if not isinstance(dep, dict):
                continue
            relation = str(dep.get("relation_type", ""))
            targets = dep.get("proposition_indices", [])
            refs = dep.get("depends_on", [])
            if not isinstance(targets, list) or not isinstance(refs, list):
                continue
            for t in targets:
                for r in refs:
                    if not isinstance(t, int) or not isinstance(r, int):
                        continue
                    if t >= n_subtasks or r >= n_subtasks or t < 0 or r < 0:
                        continue
                    if relation.startswith("after"):
                        precedence.append([f"s{r}", f"s{t}"])
                    elif relation.startswith("before"):
                        precedence.append([f"s{t}", f"s{r}"])

    if not precedence:
        precedence = [[f"s{i}", f"s{i + 1}"] for i in range(max(0, n_subtasks - 1))]

    seen: set[tuple[str, str]] = set()
    deduped: list[list[str]] = []
    for u, v in precedence:
        key = (u, v)
        if u != v and key not in seen:
            seen.add(key)
            deduped.append([u, v])
    return deduped


def _build_sync_constraints(episode: dict[str, Any], n_subtasks: int) -> list[dict[str, Any]]:
    deps = episode.get("evaluation_proposition_dependencies", [])
    sync: list[dict[str, Any]] = []
    if isinstance(deps, list):
        for dep in deps:
            if not isinstance(dep, dict):
                continue
            relation = str(dep.get("relation_type", ""))
            if relation != "while_satisfied":
                continue
            targets = dep.get("proposition_indices", [])
            refs = dep.get("depends_on", [])
            all_ids: list[str] = []
            for x in list(targets) + list(refs):
                if isinstance(x, int) and 0 <= x < n_subtasks:
                    all_ids.append(f"s{x}")
            all_ids = sorted(set(all_ids))
            if len(all_ids) >= 2:
                sync.append({"subtask_ids": all_ids, "max_lag": 1})
    return sync


def _allowed_agents(required_capabilities: set[str]) -> list[str]:
    allowed = [
        a.agent_id for a in DEFAULT_AGENTS if required_capabilities.issubset(a.capabilities)
    ]
    if not allowed:
        return [DEFAULT_AGENTS[0].agent_id]
    return allowed


def _build_candidate_assignments(
    subtasks: list[dict[str, Any]],
    precedence: list[list[str]],
    *,
    seed: int,
    horizon: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    earliest: dict[str, int] = {s["id"]: 0 for s in subtasks}
    dur: dict[str, int] = {s["id"]: int(s.get("duration_hint", 3)) for s in subtasks}

    changed = True
    while changed:
        changed = False
        for u, v in precedence:
            cand = earliest.get(u, 0) + dur.get(u, 3)
            if cand > earliest.get(v, 0):
                earliest[v] = cand
                changed = True

    candidates: list[dict[str, Any]] = []
    cid = 0
    for s in subtasks:
        sid = s["id"]
        req = set(s["required_capabilities"])
        allowed = _allowed_agents(req)
        duration = dur[sid]
        start0 = min(earliest[sid], max(0, horizon - duration))
        args = s.get("metadata", {}).get("args", {})
        resources = _extract_resource_tokens(args if isinstance(args, dict) else {})
        zone = _extract_zone(args if isinstance(args, dict) else {})

        for agent in allowed:
            for delta in (0, 1, 2):
                start = start0 + delta
                if start + duration > horizon:
                    continue
                base = float(duration) + (0.2 if agent.startswith("human") else 0.0)
                jitter = float(rng.randint(0, 2)) * 0.1
                candidates.append(
                    {
                        "candidate_id": f"c{cid}",
                        "subtask_id": sid,
                        "agent_id": agent,
                        "start": int(start),
                        "duration": int(duration),
                        "cost": round(base + jitter, 3),
                        "resources": resources,
                        "zone": zone,
                    }
                )
                cid += 1

    return candidates


def _build_conflicts_from_subtasks(
    subtasks: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    resource_to_subtasks: dict[str, set[str]] = defaultdict(set)
    zone_to_subtasks: dict[str, set[str]] = defaultdict(set)

    for s in subtasks:
        sid = s["id"]
        args = s.get("metadata", {}).get("args", {})
        if not isinstance(args, dict):
            args = {}
        for r in _extract_resource_tokens(args):
            resource_to_subtasks[r].add(sid)
        zone_to_subtasks[_extract_zone(args)].add(sid)

    resource_conflicts = [
        {"resource": r, "subtask_ids": sorted(ids)}
        for r, ids in resource_to_subtasks.items()
        if len(ids) >= 2
    ]
    spatial_conflicts = [
        {"zone": z, "subtask_ids": sorted(ids)}
        for z, ids in zone_to_subtasks.items()
        if len(ids) >= 2 and z != "global"
    ]
    return resource_conflicts, spatial_conflicts


def build_instance_from_episode(
    episode: dict[str, Any],
    *,
    episode_index: int,
    horizon: int,
    seed: int,
    source_tag: str,
) -> dict[str, Any]:
    subtasks = _build_subtasks_from_episode(episode)
    precedence = _build_precedence(episode, len(subtasks))
    sync_constraints = _build_sync_constraints(episode, len(subtasks))
    candidates = _build_candidate_assignments(
        subtasks, precedence, seed=seed + episode_index, horizon=horizon
    )
    resource_conflicts, spatial_conflicts = _build_conflicts_from_subtasks(subtasks)

    agents = [
        {"id": a.agent_id, "capabilities": sorted(a.capabilities), "metadata": {}}
        for a in DEFAULT_AGENTS
    ]

    capability_constraints = []
    for s in subtasks:
        allowed = _allowed_agents(set(s["required_capabilities"]))
        capability_constraints.append({"subtask_id": s["id"], "allowed_agents": allowed})

    durations = {s["id"]: int(s.get("duration_hint", 3)) for s in subtasks}
    costs = {c["candidate_id"]: float(c["cost"]) for c in candidates}

    ep_id = str(episode.get("episode_id", episode_index))
    instance = {
        "episode_id": ep_id,
        "task_text": str(episode.get("instruction", "unknown instruction")),
        "agents": agents,
        "subtasks": subtasks,
        "precedence": precedence,
        "candidate_assignments": candidates,
        "capability_constraints": capability_constraints,
        "resource_conflicts": resource_conflicts,
        "spatial_conflicts": spatial_conflicts,
        "sync_constraints": sync_constraints,
        "durations": durations,
        "costs": costs,
        "metadata": {
            "source": source_tag,
            "horizon": horizon,
            "n_subtasks": len(subtasks),
            "n_candidates": len(candidates),
        },
    }
    return instance


def extract_instances_from_partnr_dataset(
    dataset_path: str | Path,
    output_dir: str | Path,
    *,
    max_instances: int = 50,
    horizon: int = 40,
    seed: int = 7,
) -> list[Path]:
    raw = load_json_auto(dataset_path)
    episodes = _flatten_episodes(raw)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for idx, ep in enumerate(episodes[:max_instances]):
        inst = build_instance_from_episode(
            ep,
            episode_index=idx,
            horizon=horizon,
            seed=seed,
            source_tag=f"partnr:{Path(dataset_path).name}",
        )
        out_path = out_dir / f"instance_{idx:04d}_{inst['episode_id']}.json"
        dump_json(out_path, inst)
        written.append(out_path)
    return written


def generate_synthetic_oracle_instances(
    output_dir: str | Path,
    *,
    num_instances: int = 50,
    min_subtasks: int = 6,
    max_subtasks: int = 14,
    horizon: int = 40,
    seed: int = 11,
) -> list[Path]:
    rng = random.Random(seed)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    verbs = ["navigate_to", "pick", "place", "open", "close", "inspect"]
    objs = ["cup", "plate", "book", "bottle", "box", "towel", "pan", "spoon"]
    rooms = ["kitchen", "living_room", "bedroom", "hall"]

    for i in range(num_instances):
        n_sub = rng.randint(min_subtasks, max_subtasks)
        props = []
        for j in range(n_sub):
            fn = verbs[rng.randint(0, len(verbs) - 1)]
            obj = objs[rng.randint(0, len(objs) - 1)] + f"_{rng.randint(0, 3)}"
            room = rooms[rng.randint(0, len(rooms) - 1)]
            props.append(
                {
                    "function_name": fn,
                    "args": {"object_handles": [obj], "room": room},
                }
            )

        deps = []
        for j in range(n_sub - 1):
            deps.append(
                {
                    "proposition_indices": [j + 1],
                    "depends_on": [j],
                    "relation_type": "after_satisfied",
                    "dependency_mode": "all",
                }
            )
        if n_sub >= 4 and rng.random() < 0.5:
            deps.append(
                {
                    "proposition_indices": [1, 2],
                    "depends_on": [0],
                    "relation_type": "while_satisfied",
                    "dependency_mode": "any",
                }
            )

        episode = {
            "episode_id": f"synthetic_{i:04d}",
            "instruction": f"Synthetic cooperative task {i}",
            "evaluation_propositions": props,
            "evaluation_proposition_dependencies": deps,
        }
        inst = build_instance_from_episode(
            episode,
            episode_index=i,
            horizon=horizon,
            seed=seed,
            source_tag="synthetic_oracle_fallback",
        )
        out_path = out_dir / f"instance_{i:04d}_{inst['episode_id']}.json"
        dump_json(out_path, inst)
        written.append(out_path)

    return written

