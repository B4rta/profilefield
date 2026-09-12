from pathlib import Path

import pytest

from profilefield.artifacts import RunArtifacts, validate_run_directory


@pytest.mark.parametrize("mutation", ["metric", "config", "added", "removed_seal"])
def test_sealed_artifacts_detect_post_finalization_changes(tmp_path: Path, mutation: str) -> None:
    run = RunArtifacts.create(tmp_path, "integrity", 1)
    run.initialize({"experiment": {"name": "test"}}, {"source": "unit"}, {"train": ["a"]}, tmp_path, 1)
    run.write_table("metrics_profile", [{"metric": "rmse", "value": 2.}])
    run.finalize(0.)
    assert validate_run_directory(run.path)["integrity"] == "sha256_all_artifacts"
    if mutation == "metric":
        run.write_table("metrics_profile", [{"metric": "rmse", "value": 1.}])
    elif mutation == "config":
        (run.path / "config.yaml").write_text("changed: true\n", encoding="utf-8")
    elif mutation == "added":
        run.write_json("diagnostics/new.json", {})
    else:
        (run.path / "artifact_integrity.json").unlink()
    with pytest.raises(RuntimeError):
        validate_run_directory(run.path)
