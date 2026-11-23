import numpy as np
import pandas as pd
import pytest

from cpmaudit.datasets import FEATURES, make_clinical_risk_data


@pytest.fixture(scope="session")
def data():
    return make_clinical_risk_data(n_stays=1600, prevalence=0.15, random_state=11)


@pytest.fixture(scope="session")
def split(data):
    dev = data[data["site_id"] <= 3].reset_index(drop=True)
    ext = data[data["site_id"] > 3].reset_index(drop=True)
    return dev, ext


@pytest.fixture(scope="session")
def features():
    return FEATURES


@pytest.fixture(scope="session")
def small_result(split, features):
    from cpmaudit import StabilityAudit

    dev, ext = split
    return StabilityAudit("logistic", n_bootstrap=12, bootstrap="cluster", thresholds=(0.1, 0.2), random_state=5, keep_models=True).run(
        dev[features], dev["outcome"], groups=dev["patient_id"], subgroups=dev["sex"], ids=dev["stay_id"],
        X_external=ext[features], y_external=ext["outcome"], subgroups_external=ext["sex"],
    )


@pytest.fixture
def numeric_xy():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(600, 4))
    lp = -1.5 + X @ np.array([0.8, -0.5, 0.3, 0.0])
    y = (rng.random(600) < 1 / (1 + np.exp(-lp))).astype(int)
    return pd.DataFrame(X, columns=list("abcd")), y
