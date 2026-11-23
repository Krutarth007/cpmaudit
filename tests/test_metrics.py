import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from cpmaudit.calibration import calibration_curve, calibration_error_summary, rcs_basis
from cpmaudit.metrics import (
    brier_score,
    calibration_intercept_slope,
    fit_logistic,
    net_benefit,
    observed_expected_ratio,
    performance_summary,
    roc_auc,
    threshold_metrics,
)


@pytest.fixture
def calibrated():
    rng = np.random.default_rng(1)
    p = 1 / (1 + np.exp(-rng.normal(-1.5, 1.2, 20000)))
    y = (rng.random(p.size) < p).astype(int)
    return y, p


def test_auc_matches_sklearn(calibrated):
    y, p = calibrated
    assert roc_auc(y, p) == pytest.approx(roc_auc_score(y, p), abs=1e-12)
    p_tied = np.round(p, 1)
    assert roc_auc(y, p_tied) == pytest.approx(roc_auc_score(y, p_tied), abs=1e-12)


def test_calibration_of_true_risks(calibrated):
    y, p = calibrated
    cal = calibration_intercept_slope(y, p)
    assert abs(cal["calibration_intercept"]) < 0.05
    assert abs(cal["calibration_slope"] - 1) < 0.05
    assert observed_expected_ratio(y, p) == pytest.approx(1, abs=0.03)
    assert calibration_error_summary(y, p)["ici"] < 0.01


def test_miscalibration_is_detected(calibrated):
    y, p = calibrated
    lp = np.log(p / (1 - p))
    over = 1 / (1 + np.exp(-(2 * lp)))
    assert calibration_intercept_slope(y, over)["calibration_slope"] == pytest.approx(0.5, abs=0.05)
    assert calibration_error_summary(y, over)["ici"] > 0.02


def test_fit_logistic_matches_sklearn():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(2000, 2))
    y = (rng.random(2000) < 1 / (1 + np.exp(-(0.3 + X @ [1.0, -0.7])))).astype(int)
    ours = fit_logistic(np.column_stack([np.ones(2000), X]), y)
    sk = LogisticRegression(C=np.inf, max_iter=1000).fit(X, y)
    assert np.allclose(ours, np.r_[sk.intercept_, sk.coef_.ravel()], atol=1e-4)


def test_net_benefit_by_hand():
    y = np.array([1, 0, 1, 0, 0])
    p = np.array([0.9, 0.6, 0.2, 0.1, 0.05])
    # t = 0.5: TP=1, FP=1 -> 1/5 - 1/5 * 1 = 0
    assert net_benefit(y, p, [0.5])[0] == pytest.approx(0.0)
    m = threshold_metrics(y, p, 0.15)
    assert m["sensitivity"] == 1.0 and m["flagged_fraction"] == pytest.approx(0.6)
    assert brier_score([1, 0], [1, 0]) == 0


def test_performance_summary_keys(calibrated):
    y, p = calibrated
    out = performance_summary(y, p, thresholds=[0.1])
    for key in ("auroc", "brier", "scaled_brier", "calibration_slope", "ici", "net_benefit@0.1"):
        assert key in out


def test_rcs_and_curve(calibrated):
    y, p = calibrated
    basis = rcs_basis(np.linspace(-3, 3, 50), np.array([-2, -1, 0, 1, 2]))
    assert basis.shape == (50, 4)
    curve = calibration_curve(y, p, grid=np.array([0.0005, 0.2, 0.9999]))
    assert np.isnan(curve["observed"][0]) and np.isnan(curve["observed"][-1])
    assert curve["observed"][1] == pytest.approx(0.2, abs=0.03)
    for method in ("binned",):
        assert np.isfinite(calibration_curve(y, p, method=method)["observed"]).any()
