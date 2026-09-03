from __future__ import annotations

import numpy as np
import pytest

from profilefield.data.preprocessing import TrainOnlyStandardizer


def test_preprocessor_records_only_training_ids() -> None:
    values = np.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 8.0]])
    scaler = TrainOnlyStandardizer().fit(values[:2], ["train-a", "train-b"])
    scaler.assert_fit_on({"train-a", "train-b"})
    assert scaler.mean_ is not None
    assert np.allclose(scaler.mean_, [2.0, 3.0])
    with pytest.raises(AssertionError, match="non-training"):
        scaler.assert_fit_on({"train-a"})


def test_standardizer_round_trip() -> None:
    values = np.asarray([[1.0, 2.0], [3.0, 6.0], [5.0, 10.0]])
    scaler = TrainOnlyStandardizer().fit(values, ["a", "b", "c"])
    assert np.allclose(scaler.inverse_transform(scaler.transform(values)), values)
