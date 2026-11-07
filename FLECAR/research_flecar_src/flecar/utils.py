from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    ensure_dir(path.parent)
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                fields.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def list_instance_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]

    preferred = sorted(p for p in path.glob("instance_*.json") if p.is_file())
    if preferred:
        return preferred

    # Fallback: include json files that look like instances.
    out: list[Path] = []
    for p in sorted(path.glob("*.json")):
        if not p.is_file():
            continue
        stem = p.stem.lower()
        if "manifest" in stem or "summary" in stem:
            continue
        out.append(p)
    return out
