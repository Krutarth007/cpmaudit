import os

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from cpmaudit import StabilityAudit, StabilityResult, stability_audit


def test_shapes_and_columns(small_result, split):
    dev, ext = split
    r = small_result
    assert r.P.shape == (12, len(dev)) and r.P_external.shape == (12, len(ext))
    ind = r.individual()
    for col in ("id", "subgroup", "outcome", "risk", "risk_lower", "risk_upper", "mape", "cii@0.1", "decision@0.2"):
        assert col in ind
    assert ind["cii@0.1"].between(0, 1).all()
    assert (ind["risk_lower"] <= ind["risk_upper"]).all()
    assert r.instability_index() == pytest.approx(ind["mape"].mean())


def test_summary_contains_both_populations(small_result):
    s = small_result.summary()
    assert set(s["population"]) == {"development", "external"}
    assert {"instability_index", "cii_mean@0.1", "share_flip_prone@0.2"} <= set(s["measure"])


def test_cii_definition(small_result):
    r = small_result
    j = int(np.argmax(r.individual()["cii@0.1"].to_numpy()))
    expected = np.mean((r.P[:, j] >= 0.1) != (r.p_original[j] >= 0.1))
    assert r.individual()["cii@0.1"].iloc[j] == pytest.approx(expected)


def test_tables(small_result):
    r = small_result
    opt = r.optimism().set_index("metric")
    assert {"optimism_corrected", "estimate_632", "estimate_632plus"} <= set(opt.columns)
    assert np.isfinite(opt.loc["auroc", "estimate_632plus"])
    assert len(r.subgroup_summary()) == 2
    assert len(r.subgroup_summary("external")) == 2
    assert {"original_model", "replicates_lower"} <= set(r.performance_variability("external").columns)
    assert len(r.distribution_stability()) == 5
    cal = r.calibration_instability()
    assert cal["replicates"].shape[0] == r.n_replicates
    dc = r.decision_curve_instability(thresholds=[0.05, 0.1])
    assert set(dc["strategy"]) == {"original", "treat_all", "treat_none", "replicate"}


def test_reproducible_and_parallel_equivalent(split, features):
    dev, _ = split
    kw = dict(n_bootstrap=6, random_state=99, thresholds=(0.1,))
    a = StabilityAudit("logistic_l2", **kw).run(dev[features], dev["outcome"])
    b = StabilityAudit("logistic_l2", n_jobs=2, **kw).run(dev[features], dev["outcome"])
    assert np.allclose(a.P, b.P)
    assert np.allclose(a.p_original, b.p_original)


def test_external_without_outcomes(split, features):
    dev, ext = split
    r = StabilityAudit("logistic", n_bootstrap=5, random_state=1).run(dev[features], dev["outcome"], X_external=ext[features])
    assert r.y_external is None
    assert "external" in set(r.summary()["population"])
    with pytest.raises(ValueError):
        r.calibration_instability("external")


def test_predict_interval_and_flags(small_result, split, features):
    _, ext = split
    out = small_result.predict_interval(ext[features].head(10))
    assert len(out) == 10 and (out["risk_lower"] <= out["risk_upper"]).all()
    mask = small_result.flag_unstable(max_interval_width=0.05, max_cii=0.2)
    assert mask.dtype == bool and mask.size == small_result.P.shape[1]


def test_callable_and_estimator_specs(numeric_xy):
    X, y = numeric_xy

    def build(Xb, yb, random_state=None):
        return LogisticRegression(max_iter=500).fit(Xb, yb)

    r1 = stability_audit(build, X, y, n_bootstrap=4, random_state=0)
    r2 = stability_audit(LogisticRegression(max_iter=500), X.to_numpy(), y, n_bootstrap=4, random_state=0, bootstrap="stratified")
    assert r1.n_replicates == 4 and r2.n_replicates == 4


def test_predict_fn_for_custom_objects(numeric_xy):
    X, y = numeric_xy

    class Linear:
        def __init__(self, coef):
            self.coef = coef

    def build(Xb, yb):
        coef = np.linalg.lstsq(np.c_[np.ones(len(Xb)), Xb], yb, rcond=None)[0]
        return Linear(coef)

    def predict(model, Xn):
        return np.clip(np.c_[np.ones(len(Xn)), Xn] @ model.coef, 0.001, 0.999)

    r = StabilityAudit(build, n_bootstrap=3, predict_fn=predict, smooth_calibration=False).run(X, y)
    assert r.n_replicates == 3


def test_failures_are_recorded_and_limited(numeric_xy):
    X, y = numeric_xy
    calls = {"n": 0}

    def flaky(Xb, yb):
        calls["n"] += 1
        if calls["n"] in (3,):
            raise RuntimeError("boom")
        return LogisticRegression(max_iter=300).fit(Xb, yb)

    r = StabilityAudit(flaky, n_bootstrap=10, max_failure_fraction=0.2).run(X, y)
    assert r.n_replicates == 9 and len(r.failures) == 1

    def always_fail(Xb, yb):
        raise RuntimeError("nope")

    with pytest.raises(RuntimeError):
        StabilityAudit(always_fail, n_bootstrap=5).run(X, y)


def test_input_errors(numeric_xy):
    X, y = numeric_xy
    with pytest.raises(ValueError):
        StabilityAudit("logistic", bootstrap="cluster", n_bootstrap=3).run(X, y)
    with pytest.raises(ValueError):
        StabilityAudit("logistic", n_bootstrap=3).run(X, np.zeros(len(y)))
    with pytest.raises(ValueError):
        StabilityAudit("not-a-model")
    with pytest.raises(ValueError):
        StabilityAudit("logistic", n_bootstrap=3).run(X, y, y_external=y)


def test_save_load_export(small_result, tmp_path):
    path = tmp_path / "res.joblib"
    small_result.save(str(path))
    loaded = StabilityResult.load(str(path))
    assert np.allclose(loaded.P, small_result.P)
    paths = small_result.export(str(tmp_path / "out"))
    for name in ("summary.csv", "individual_development.csv", "individual_external.csv", "result.json", "report.html", "optimism.csv"):
        assert name in paths and os.path.exists(paths[name])
    js = small_result.to_dict()
    assert js["config"]["n_bootstrap_effective"] == 12
    frame = pd.read_csv(paths["summary.csv"])
    assert len(frame) > 10
