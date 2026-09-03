from __future__ import annotations

from pathlib import Path

import pytest

from profilefield.baselines.geovae_sgs import published_metric_rows, verify_baseline

ROOT = Path(__file__).resolve().parents[1]


def test_pinned_baseline_hashes_are_unchanged() -> None:
    result = verify_baseline(ROOT, "manifests/geovae_sgs_baseline.json")
    assert result["verified"] is True
    assert result["commit"] == "7cdf7d9690b0e8a08b103074d2826d9391119ffc"


def test_published_independent_gp_metrics_recompute() -> None:
    profile, calibration = published_metric_rows(
        ROOT / "third_party/geovae-sgs/validations/reviewer_analysis/combined_prediction_pairs.csv"
    )
    values = {row["metric"]: row["value"] for row in profile if row["model"] == "GeoVAE-GP-KC"}
    assert values["n"] == 3886
    assert values["mae"] == pytest.approx(0.48899781836873973)
    assert values["rmse"] == pytest.approx(0.6304096493318094)
    assert values["bias"] == pytest.approx(0.00685775366551461)
    assert len(calibration) == 4
