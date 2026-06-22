"""Experiment run directory and metadata helpers."""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPERIMENT_METADATA_VERSION = 1


@dataclass(frozen=True)
class ExperimentRun:
    """Filesystem layout for one experiment/evaluation/training run."""

    root: Path
    metadata_path: Path
    artifacts_dir: Path
    checkpoints_dir: Path
    logs_dir: Path


def create_experiment_run(
    *,
    root: str | Path,
    name: str,
    kind: str,
    config: dict[str, Any] | None = None,
    tags: list[str] | None = None,
    timestamp: str | None = None,
) -> ExperimentRun:
    """Create a run directory with standard subfolders and metadata."""
    run_id = f"{timestamp or _timestamp()}_{sanitize_name(name)}"
    run_root = Path(root) / run_id
    artifacts_dir = run_root / "artifacts"
    checkpoints_dir = run_root / "checkpoints"
    logs_dir = run_root / "logs"

    for directory in (artifacts_dir, checkpoints_dir, logs_dir):
        directory.mkdir(parents=True, exist_ok=False)

    metadata_path = run_root / "metadata.json"
    metadata = {
        "metadata_version": EXPERIMENT_METADATA_VERSION,
        "run_id": run_id,
        "name": name,
        "kind": kind,
        "status": "running",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": config or {},
        "tags": tags or [],
        "paths": {
            "root": str(run_root),
            "artifacts": str(artifacts_dir),
            "checkpoints": str(checkpoints_dir),
            "logs": str(logs_dir),
        },
    }
    write_json(metadata_path, metadata)

    return ExperimentRun(
        root=run_root,
        metadata_path=metadata_path,
        artifacts_dir=artifacts_dir,
        checkpoints_dir=checkpoints_dir,
        logs_dir=logs_dir,
    )


def mark_experiment_status(
    run: ExperimentRun,
    status: str,
    *,
    summary: dict[str, Any] | None = None,
) -> None:
    """Update experiment metadata with a terminal or intermediate status."""
    metadata = read_json(run.metadata_path)
    metadata["status"] = status
    metadata["updated_at"] = datetime.now(timezone.utc).isoformat()
    if summary is not None:
        metadata["summary"] = summary
    write_json(run.metadata_path, metadata)


def sanitize_name(name: str) -> str:
    """Return a filesystem-friendly lowercase name."""
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip()).strip("_")
    return safe.lower() or "run"


def write_json(path: str | Path, payload: Any) -> None:
    """Write a JSON document with stable formatting."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_json(path: str | Path) -> Any:
    """Read a JSON document."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

