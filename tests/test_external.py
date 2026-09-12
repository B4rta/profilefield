import numpy as np
import pandas as pd
import pytest

from profilefield.experiments.external import minimum_distance_km, select_external_cases


def test_external_selection_does_not_use_responses_or_row_order():
    frame = pd.DataFrame({"sample_id": list("abcdef"), "rh_0": range(6)})
    first = select_external_cases(frame, 3, 2026)
    changed = frame.iloc[::-1].copy()
    changed["rh_0"] = 10000
    assert first.sample_id.tolist() == select_external_cases(changed, 3, 2026).sample_id.tolist()
    with pytest.raises(ValueError, match="unique"):
        select_external_cases(pd.concat([frame, frame]), 3, 2026)


def test_great_circle_distance_detects_overlap_and_continental_separation():
    source = np.array([[-53.0, 3.0]])
    assert minimum_distance_km(source, source) == 0
    assert minimum_distance_km(source, np.array([[175.85, -37.35]])) > 10000
