import numpy as np
import pytest

from cpmaudit import MultiplicityAudit, multiplicity_from_predictions, multiplicity_measures, param_grid_variants, seed_variants
from cpmaudit.multiplicity import rashomon_set


def test_measures_by_hand():
    P = np.array([[0.05, 0.2, 0.3, 0.6], [0.15, 0.2, 0.05, 0.6], [0.05, 0.25, 0.3, 0.7]])
    m = multiplicity_measures(P, [0.1], reference=0)
    # decisions: ref [0,1,1,1]; model1 [1,1,0,1]; model2 [0,1,1,1]
    assert m["ambiguity@0.1"] == pytest.approx(0.5)
    assert m["discrepancy@0.1"] == pytest.approx(0.5)
    assert m["vpr_mean"] == pytest.approx(np.mean([0.10, 0.05, 0.25, 0.1]))


def test_rashomon_set():
    assert rashomon_set([0.80, 0.795, 0.70], 0.01).tolist() == [True, True, False]
    assert rashomon_set([0.30, 0.305, 0.40], 0.01, higher_is_better=False).tolist() == [True, True, False]


def test_audit_runs(numeric_xy):
    X, y = numeric_xy
    res = MultiplicityAudit(seed_variants("random_forest", 4), epsilon=0.05, thresholds=(0.2,)).run(X[:400], y[:400], X[400:], y[400:])
    assert res.candidates["in_rashomon_set"].sum() >= 1
    assert "ambiguity@0.2" in set(res.summary()["measure"])
    assert len(res.individual()) == len(X) - 400


def test_param_grid_and_from_predictions(numeric_xy):
    from sklearn.linear_model import LogisticRegression

    X, y = numeric_xy
    cands = param_grid_variants(LogisticRegression(max_iter=500), {"C": [0.01, 1.0, 100.0]})
    assert len(cands) == 3
    res = MultiplicityAudit(cands, epsilon=0.02).run(X[:400], y[:400], X[400:], y[400:], X_target=X.head(20))
    assert len(res.individual()) == 20
    r = multiplicity_from_predictions(np.random.default_rng(0).random((5, 30)), scores=[0.7, 0.71, 0.6, 0.705, 0.5], epsilon=0.02)
    assert r.P.shape[0] == 3


def test_requires_two_candidates():
    with pytest.raises(ValueError):
        MultiplicityAudit(["logistic"])
