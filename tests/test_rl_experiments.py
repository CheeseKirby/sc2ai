from __future__ import annotations

import pytest

from rl.experiments import (
    create_experiment_run,
    mark_experiment_status,
    read_json,
    sanitize_name,
)


@pytest.mark.unit
def test_sanitize_name_is_filesystem_friendly() -> None:
    assert sanitize_name(" Imitation Run: Easy/Medium ") == "imitation_run_easy_medium"
    assert sanitize_name("!!!") == "run"


@pytest.mark.unit
def test_create_experiment_run_writes_standard_layout(tmp_path) -> None:
    run = create_experiment_run(
        root=tmp_path,
        name="Imitation Smoke",
        kind="imitation",
        config={"epochs": 3},
        tags=["unit"],
        timestamp="20260618_120000",
    )

    assert run.root.name == "20260618_120000_imitation_smoke"
    assert run.artifacts_dir.is_dir()
    assert run.checkpoints_dir.is_dir()
    assert run.logs_dir.is_dir()

    metadata = read_json(run.metadata_path)
    assert metadata["metadata_version"] == 1
    assert metadata["name"] == "Imitation Smoke"
    assert metadata["kind"] == "imitation"
    assert metadata["status"] == "running"
    assert metadata["config"] == {"epochs": 3}
    assert metadata["tags"] == ["unit"]


@pytest.mark.unit
def test_mark_experiment_status_updates_metadata(tmp_path) -> None:
    run = create_experiment_run(
        root=tmp_path,
        name="Eval",
        kind="evaluation",
        timestamp="20260618_120000",
    )

    mark_experiment_status(run, "complete", summary={"victories": 1})

    metadata = read_json(run.metadata_path)
    assert metadata["status"] == "complete"
    assert metadata["summary"] == {"victories": 1}
    assert "updated_at" in metadata

