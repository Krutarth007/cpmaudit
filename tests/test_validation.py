import numpy as np
import pandas as pd
import pytest

from cpmaudit._validation import as_binary_outcome, as_probability, as_thresholds, check_features, take_rows


def test_binary_outcome_accepts_common_types():
    assert as_binary_outcome([0, 1, 1]).tolist() == [0, 1, 1]
    assert as_binary_outcome(pd.Series([True, False])).tolist() == [1, 0]
    assert as_binary_outcome(np.array([0.0, 1.0])).dtype == np.int8


@pytest.mark.parametrize("bad", [[0, 2], [0, np.nan], ["a", "b"]])
def test_binary_outcome_rejects_invalid(bad):
    with pytest.raises(ValueError):
        as_binary_outcome(bad)


def test_probability_range():
    with pytest.raises(ValueError):
        as_probability([0.2, 1.2])
    assert as_probability([0, 1], clip=True).min() > 0


def test_thresholds():
    assert as_thresholds(0.1) == (0.1,)
    with pytest.raises(ValueError):
        as_thresholds([0, 0.5])


def test_take_rows_keeps_dataframe():
    df = pd.DataFrame({"a": [1, 2, 3]}, index=[10, 20, 30])
    out = take_rows(df, np.array([2, 0]))
    assert list(out["a"]) == [3, 1]
    assert check_features([1, 2, 3]).shape == (3, 1)
