"""Paired effects with an explicit separation of repetitions and sampling units."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import binomtest

from profilefield.evaluation.bootstrap import block_bootstrap_interval


def pair_model_values(
    frame: pd.DataFrame,
    *,
    proposed_model: str,
    comparator_model: str,
    keys: Sequence[str],
) -> pd.DataFrame:
    """Join one endpoint per key without silent attrition or Cartesian duplication.

    Callers must include all endpoint and sampling identifiers in ``keys``. A
    missing model, duplicate key, or unmatched observation is an error. Values
    are kept unchanged; callers must explicitly account for any nonfinite rows.
    """
    key_columns = list(keys)
    if not key_columns or len(set(key_columns)) != len(key_columns):
        raise ValueError("keys must be nonempty and unique")
    if {"model", "value"}.intersection(key_columns):
        raise ValueError("keys cannot include model or value")
    if proposed_model == comparator_model:
        raise ValueError("The proposed and comparator models must differ")
    required = {"model", "value", *key_columns}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing pairing columns: {sorted(required - set(frame.columns))}")
    comparator = frame.loc[frame["model"] == comparator_model, [*key_columns, "value"]]
    proposed = frame.loc[frame["model"] == proposed_model, [*key_columns, "value"]]
    if comparator.empty or proposed.empty:
        raise ValueError("Both models must have observations for pairing")
    for name, values in ((comparator_model, comparator), (proposed_model, proposed)):
        if values.duplicated(key_columns).any():
            raise ValueError(f"Duplicate observation keys for {name}")
    paired = comparator.rename(columns={"value": "comparator"}).merge(
        proposed.rename(columns={"value": "proposed"}),
        on=key_columns,
        how="outer",
        validate="one_to_one",
        indicator=True,
        sort=True,
    )
    if not (paired["_merge"] == "both").all():
        raise ValueError("Unmatched observations between the paired models")
    return paired.drop(columns="_merge")


def paired_unit_effects(
    frame: pd.DataFrame,
    *,
    proposed_model: str,
    comparator_model: str,
    unit_columns: Sequence[str],
    repetition_columns: Sequence[str] = ("seed",),
) -> pd.DataFrame:
    """Average paired repetitions within units, never counting seeds as new sites.

    ``frame`` must contain one selected endpoint. Units receive equal weight,
    regardless of their size. The result is a mean of unit-level endpoint
    differences, not a pooled RMSE. Every unit must have the same repetition
    identifiers, and each model must contain exactly the same observations.
    Choosing a geographic unit does not by itself establish independence.
    """
    units, repetitions = list(unit_columns), list(repetition_columns)
    if not units or set(units).intersection(repetitions):
        raise ValueError("unit_columns must be nonempty and distinct from repetitions")
    paired = pair_model_values(
        frame,
        proposed_model=proposed_model,
        comparator_model=comparator_model,
        keys=[*units, *repetitions],
    )
    if paired[[*units, *repetitions]].isna().any().any():
        raise ValueError("Sampling-unit and repetition identifiers cannot be missing")
    for column in ("comparator", "proposed"):
        paired[column] = pd.to_numeric(paired[column], errors="raise")
        if not np.isfinite(paired[column].to_numpy(float)).all():
            raise ValueError("Paired endpoint values must be finite")
    if repetitions:
        expected = set(paired[repetitions].itertuples(index=False, name=None))
        for _, group in paired.groupby(units, dropna=False):
            if set(group[repetitions].itertuples(index=False, name=None)) != expected:
                raise ValueError("All units must have the same repetition identifiers")
    paired["difference"] = paired["proposed"] - paired["comparator"]
    return (
        paired.groupby(units, dropna=False, sort=True)
        .agg(
            comparator=("comparator", "mean"),
            proposed=("proposed", "mean"),
            difference=("difference", "mean"),
            n_repetitions=("difference", "size"),
        )
        .reset_index()
    )


def summarize_paired_units(
    differences: NDArray[np.float64],
    *,
    independent_units: bool = False,
    confidence: float = 0.95,
    replicates: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """Summarize equal-unit effects; inferential quantities are opt-in.

    The default is descriptive and returns unavailable intervals and p-values.
    ``independent_units=True`` asserts independent, exchangeable sampling units;
    repeated fitted models or overlapping cross-validation folds do not meet
    that requirement simply because their numeric identifiers differ. The
    percentile interval targets the mean unit effect. The two-sided exact sign
    test instead targets a probability of positive differences equal to 1/2,
    conditional on nonzero effects; it does not test zero mean. See SciPy's
    ``binomtest`` documentation. No spatial independence is inferred here.
    """
    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("differences must be a nonempty finite one-dimensional array")
    if not isinstance(independent_units, (bool, np.bool_)):
        raise ValueError("independent_units must be an explicit boolean")
    nonzero = values[values != 0.0]
    result: dict[str, Any] = {
        "n_units": len(values),
        "mean_difference": float(np.mean(values)),
        "median_difference": float(np.median(values)),
        "fraction_proposed_lower": float(np.mean(values < 0.0)),
        "n_nonzero_units": len(nonzero),
        "independent_units_assumed": bool(independent_units),
        "ci_lower": float("nan"),
        "ci_upper": float("nan"),
        "p_value_sign": float("nan"),
        "inference_scope": "descriptive_only",
    }
    if independent_units:
        _, lower, upper = block_bootstrap_interval(
            values,
            np.arange(len(values), dtype=np.int64),
            confidence=confidence,
            replicates=replicates,
            seed=seed,
        )
        result.update(
            ci_lower=lower,
            ci_upper=upper,
            p_value_sign=(
                float(binomtest(int(np.count_nonzero(nonzero > 0.0)), len(nonzero)).pvalue)
                if len(nonzero)
                else 1.0
            ),
            inference_scope="conditional_on_independent_exchangeable_units",
        )
    return result
