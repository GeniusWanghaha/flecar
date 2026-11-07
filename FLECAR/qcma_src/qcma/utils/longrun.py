"""Long-run experiment helpers: checkpoint/resume, heartbeat, and partial aggregation."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from qcma.utils.io import dump_json


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def append_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    fieldnames = list(rows[0].keys())
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def row_key(row: dict[str, Any], fields: list[str]) -> tuple[str, ...]:
    return tuple(str(row.get(k, "")) for k in fields)


def load_done_keys(path: Path, fields: list[str]) -> set[tuple[str, ...]]:
    rows = read_csv_rows(path)
    return {row_key(r, fields) for r in rows}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_heartbeat(
    *,
    path: Path,
    job_id: str,
    status: str,
    done: int,
    total: int,
    last_item: str,
    notes: str,
) -> None:
    payload = {
        "job_id": job_id,
        "status": status,
        "done": done,
        "total": total,
        "progress": round(done / total, 4) if total > 0 else 0.0,
        "last_item": last_item,
        "notes": notes,
        "timestamp_utc": utc_now_iso(),
    }
    dump_json(path, payload)
