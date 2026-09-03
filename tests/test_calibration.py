from __future__ import annotations

import numpy as np
import pytest

from profilefield.calibration.scaling import JointVarianceScaleCalibrator, VarianceScaleCalibrator


def test_calibration_ids_are_isolated_from_test() -> None:
    truth = np.asarray([[1.0, 2.0], [2.0, 3.0]])
    mean = truth + 0.2
    variance = np.ones_like(truth)
    calibrator = VarianceScaleCalibrator().fit(truth, mean, variance, ["cal-a", "cal-b"])
    calibrator.assert_isolated_from({"test-a", "test-b"})
    with pytest.raises(AssertionError, match="Calibration/test overlap"):
        calibrator.assert_isolated_from({"cal-b", "test-a"})
    assert np.all(calibrator.transform(variance) > 0)


def test_joint_variance_scaler_uses_one_profile_temperature() -> None:
    truth = np.array([[1.0, 3.0], [2.0, 5.0]])
    mean = np.array([[0.0, 1.0], [1.0, 3.0]])
    variance = np.ones_like(truth)
    calibrator = JointVarianceScaleCalibrator().fit(truth, mean, variance, ["cal-a", "cal-b"])
    assert calibrator.scale_ == pytest.approx(2.5)
    assert np.allclose(calibrator.transform(variance), 2.5)
    calibrator.assert_isolated_from({"test-a"})
