from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

from profilefield.artifacts import RunArtifacts, validate_run_directory


def test_run_artifact_contract_and_hashes() -> None:
    output = Path.home() / ".codex/test-output" / uuid.uuid4().hex[:8]
    try:
        artifacts = RunArtifacts.create(output, "t", 4, run_id="r")
        artifacts.initialize(
            {"experiment": {"name": "test"}},
            {"source": "unit"},
            {"train": ["a"], "test": ["b"]},
            Path.cwd(),
            4,
        )
        artifacts.finalize(0.1)
        result = validate_run_directory(artifacts.path)
        assert result["status"] == "valid"
        with (artifacts.path / "run_metadata.json").open("r", encoding="utf-8") as handle:
            assert json.load(handle)["seed"] == 4
    finally:
        if output.exists():
            shutil.rmtree(output)
