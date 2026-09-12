from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from profilefield.evaluation.bootstrap import block_bootstrap_interval
from profilefield.evaluation.inference import (
    pair_model_values,
    paired_unit_effects,
    summarize_paired_units,
)


def _paired_rows(seeds: tuple[int, ...] = (1, 2)) -> pd.DataFrame:
    rows = []
    for block, effect in ((10, -2.0), (20, 1.0)):
        for seed in seeds:
            for model, value in (("reference", 10.0 + seed), ("candidate", 10.0 + seed + effect)):
                rows.append({"block": block, "seed": seed, "model": model, "value": value})
    return pd.DataFrame(rows)


def _effects(frame: pd.DataFrame) -> pd.DataFrame:
    return paired_unit_effects(
        frame,
        proposed_model="candidate",
        comparator_model="reference",
        unit_columns=("block",),
    )


def test_algorithm_repetition_does_not_manufacture_geographic_replication() -> None:
    original = _effects(_paired_rows())
    repeated = _effects(_paired_rows(tuple(range(20))))
    np.testing.assert_allclose(original["difference"], [-2.0, 1.0])
    np.testing.assert_allclose(original["difference"], repeated["difference"])
    for frame in (original, repeated):
        summary = summarize_paired_units(frame["difference"].to_numpy(float))
        assert summary["n_units"] == 2
        assert summary["mean_difference"] == -0.5
        assert np.isnan(summary["p_value_sign"])
        assert np.isnan(summary["ci_lower"])
        assert summary["inference_scope"] == "descriptive_only"


def test_pairing_rejects_duplicates_and_unmatched_values() -> None:
    frame = _paired_rows()
    with pytest.raises(ValueError, match="Duplicate observation keys"):
        _effects(pd.concat([frame, frame.iloc[[0]]]))
    with pytest.raises(ValueError, match="Unmatched observations"):
        _effects(frame.iloc[1:])


def test_pairing_rejects_unbalanced_repetitions_even_when_models_match() -> None:
    frame = _paired_rows()
    frame = frame.loc[~((frame["block"] == 10) & (frame["seed"] == 1))]
    with pytest.raises(ValueError, match="same repetition identifiers"):
        _effects(frame)


@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_nonfinite_effect_is_not_silently_removed(value: float) -> None:
    frame = _paired_rows()
    frame.loc[0, "value"] = value
    with pytest.raises(ValueError, match="must be finite"):
        _effects(frame)


def test_pairing_aligns_by_identifiers_not_input_order() -> None:
    frame = _paired_rows().sample(frac=1, random_state=9)
    np.testing.assert_allclose(_effects(frame)["difference"], [-2.0, 1.0])
    with pytest.raises(ValueError, match="must differ"):
        pair_model_values(
            frame, proposed_model="candidate", comparator_model="candidate", keys=("block", "seed")
        )


def test_exact_sign_test_keeps_five_unit_resolution_and_excludes_ties() -> None:
    result = summarize_paired_units(
        np.asarray([-1.0, -2.0, -3.0, -4.0, -5.0, 0.0]),
        independent_units=True,
        replicates=100,
    )
    assert result["p_value_sign"] == 0.0625
    assert result["n_units"] == 6
    assert result["n_nonzero_units"] == 5
    assert result["independent_units_assumed"]
    assert result["ci_lower"] <= result["mean_difference"] <= result["ci_upper"]


def test_identical_models_and_single_unit_have_explicit_degeneracy() -> None:
    tied = summarize_paired_units(np.zeros(4), independent_units=True, replicates=100)
    assert tied["p_value_sign"] == 1.0
    assert tied["ci_lower"] == tied["ci_upper"] == 0.0
    single = summarize_paired_units(np.asarray([-1.0]), independent_units=True, replicates=100)
    assert single["p_value_sign"] == 1.0
    assert np.isnan(single["ci_lower"])


def test_bootstrap_preserves_pooled_estimand_with_unequal_block_sizes() -> None:
    estimate, lower, upper = block_bootstrap_interval(
        np.asarray([0.0, 10.0, 10.0, 10.0]),
        np.asarray([0, 1, 1, 1], dtype=np.int64),
        replicates=200,
        seed=7,
    )
    assert estimate == 7.5  # Observation-weighted, not equal-block mean 5.
    assert lower == 0.0
    assert upper == 10.0


@pytest.mark.parametrize("replicates", [0, -1, 1.5, True])
def test_bootstrap_rejects_invalid_replicate_count(replicates: int) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        block_bootstrap_interval(np.ones(3), np.arange(3), replicates=replicates)


@pytest.mark.parametrize("confidence", [0.0, 1.0, np.nan, 1.1])
def test_bootstrap_rejects_invalid_confidence(confidence: float) -> None:
    with pytest.raises(ValueError, match="strictly between"):
        block_bootstrap_interval(np.ones(3), np.arange(3), confidence=confidence)


def test_bootstrap_rejects_misaligned_or_nonfinite_observations() -> None:
    with pytest.raises(ValueError, match="align"):
        block_bootstrap_interval(np.ones(3), np.arange(2))
    with pytest.raises(ValueError, match="finite"):
        block_bootstrap_interval(np.asarray([1.0, np.nan]), np.arange(2))
    with pytest.raises(ValueError, match="integer geographic"):
        block_bootstrap_interval(np.ones(2), np.asarray([0.0, np.nan]))


def test_bootstrap_is_deterministic_and_rejects_invalid_statistic() -> None:
    values = np.asarray([1.0, 2.0, 3.0, 7.0])
    identifiers = np.asarray([0, 0, 1, 2], dtype=np.int64)
    assert block_bootstrap_interval(
        values, identifiers, seed=24, replicates=100
    ) == block_bootstrap_interval(values, identifiers, seed=24, replicates=100)
    with pytest.raises(ValueError, match="finite scalar"):
        block_bootstrap_interval(values, identifiers, statistic=lambda _: float("nan"))
