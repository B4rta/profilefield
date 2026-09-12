from __future__ import annotations

import numpy as np
import pytest
from sklearn.tree import DecisionTreeRegressor

from profilefield.context.crossfit import spatial_crossfit_predictions


def test_crossfit_excludes_entire_training_block_and_does_not_mutate_estimator() -> None:
    x = np.arange(12, dtype=float)[:, None]
    y = (x**2).reshape(12, 1)
    blocks = np.repeat(np.arange(4), 3)
    model = DecisionTreeRegressor(random_state=7).fit(x, y)
    full_predictions = model.predict(x).copy()
    predicted, folds = spatial_crossfit_predictions(model, x, y, blocks, list(map(str, range(12))), 4)
    np.testing.assert_array_equal(model.predict(x), full_predictions)
    assert not np.allclose(predicted[:, 0], full_predictions)
    held_out = []
    for fold in folds:
        assert set(fold["fit_blocks"]).isdisjoint(fold["held_out_blocks"])
        assert set(fold["fit_sample_ids"]).isdisjoint(fold["held_out_sample_ids"])
        held_out.extend(fold["held_out_sample_ids"])
    assert sorted(held_out) == sorted(map(str, range(12)))


def test_crossfit_rejects_insufficient_blocks() -> None:
    with pytest.raises(ValueError, match="distinct training blocks"):
        spatial_crossfit_predictions(DecisionTreeRegressor(), np.ones((4, 1)), np.ones((4, 2)),
                                     np.zeros(4, dtype=np.int64), list("abcd"), 2)
