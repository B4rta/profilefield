import numpy as np
import pytest
import torch

from profilefield.spatial.independent_gp import IndependentSVGP
from profilefield.spatial.lmc_svgp import LMCSVGP
from profilefield.spatial.restore import restore_spatial


@pytest.mark.parametrize("kind", ["independent", "lmc"])
def test_restored_state_preserves_predictions_without_fit(kind, monkeypatch):
    rng = np.random.default_rng(4)
    x, y = rng.normal(size=(12, 2)), rng.normal(size=(12, 2))
    options = dict(output_dim=2, inducing_count=5, training_steps=3, seed=6)
    model = IndependentSVGP(**options) if kind == "independent" else LMCSVGP(**options, latent_count=2)
    model.fit(x, y, [str(i) for i in range(12)])
    state = model.checkpoint()
    torch.manual_seed(18)
    expected = model.predict(x[:5], posterior_samples=20)
    def forbidden(*args, **kwargs):
        raise AssertionError("Restoration must not fit")
    monkeypatch.setattr(type(model), "fit", forbidden)
    restored = restore_spatial(state)
    torch.manual_seed(18)
    actual = restored.predict(x[:5], posterior_samples=20)
    for field in ("mean", "variance", "samples", "full_covariance"):
        np.testing.assert_allclose(getattr(actual, field), getattr(expected, field), atol=1e-10)
