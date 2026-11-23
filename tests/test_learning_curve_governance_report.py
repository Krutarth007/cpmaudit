import pytest

from cpmaudit import StabilityPolicy, instability_curve, model_facts
from cpmaudit.report import report_to_string


def test_instability_shrinks_with_size(data, features):
    curve = instability_curve(
        "logistic", data[features], data["outcome"], sizes=[200, 900], n_bootstrap=8, n_repeats=2,
        groups=data["patient_id"], bootstrap="cluster", reference_size=400, random_state=3,
    )
    means = curve.table.groupby("size")["instability_index"].mean()
    assert means.loc[200] > means.loc[900]
    assert curve.minimum_size("instability_index", target=1.0) == 200
    assert curve.minimum_size("instability_index", target=0.0) is None
    assert not curve.aggregate().empty


def test_curve_size_check(data, features):
    with pytest.raises(ValueError):
        instability_curve("logistic", data[features], data["outcome"], sizes=[10**7], n_bootstrap=3)


def test_policy(small_result):
    lenient = StabilityPolicy(max_instability_index=1, max_mape_p90=1, max_share_wide_interval=1, max_share_flip_prone=1,
                              max_abs_calibration_slope_optimism=10, min_replicates=10, populations=["development", "external"])
    rep = lenient.evaluate(small_result)
    assert rep.passed and rep.failed().empty
    rep.raise_if_failed()
    strict = StabilityPolicy(max_instability_index=1e-6, min_replicates=10)
    rep2 = strict.evaluate(small_result)
    assert not rep2.passed
    with pytest.raises(AssertionError):
        rep2.raise_if_failed()


def test_model_facts_and_report(small_result):
    facts = model_facts(small_result)
    assert facts["replicates"] == 12 and "0.1" in facts["thresholds"]
    html = report_to_string(small_result, title="Audit")
    assert "<title>Audit</title>" in html and "Stability summary" in html


def test_report_figures(small_result):
    pytest.importorskip("matplotlib")
    assert "data:image/png;base64" in report_to_string(small_result)
