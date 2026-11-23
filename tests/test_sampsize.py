import pytest

from cpmaudit import cox_snell_r2_from_c, max_cox_snell_r2, minimum_sample_size_binary


def test_reference_example():
    # prevalence 0.139, 24 parameters, anticipated Cox-Snell R2 0.051
    res = minimum_sample_size_binary(24, 0.139, r2_cox_snell=0.051)
    sizes = res.criteria["sample_size"].tolist()
    assert sizes == [4115, 833, 184]
    assert res.minimum_n == 4115
    assert res.max_r2_cox_snell == pytest.approx(0.5535, abs=1e-3)
    assert res.criteria.loc[1, "shrinkage"] == pytest.approx(0.648, abs=1e-3)


def test_r2_from_c_monotone():
    low = cox_snell_r2_from_c(0.7, 0.1, n_sim=200_000)
    high = cox_snell_r2_from_c(0.85, 0.1, n_sim=200_000)
    assert 0 < low < high < max_cox_snell_r2(0.1)


def test_input_checks():
    with pytest.raises(ValueError):
        minimum_sample_size_binary(10, 0.1)
    with pytest.raises(ValueError):
        minimum_sample_size_binary(10, 0.1, r2_cox_snell=0.9)
    with pytest.raises(ValueError):
        cox_snell_r2_from_c(0.4, 0.1)
    res = minimum_sample_size_binary(10, 0.2, r2_nagelkerke=0.15)
    assert res.r2_nagelkerke == pytest.approx(0.15)
