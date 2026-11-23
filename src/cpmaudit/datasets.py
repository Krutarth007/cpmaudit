"""Synthetic clinical risk data for examples, tests and teaching.

The generator mimics the structure that matters for instability audits:
several sites with different case mix and measurement practice, patients
with repeated stays, missing measurements, non-linear risk relations and a
known true risk for every row. No real patient data are involved.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from ._validation import expit


def make_clinical_risk_data(
    n_stays: int = 5000,
    prevalence: float = 0.12,
    n_sites: int = 5,
    stays_per_patient: float = 1.25,
    missing_rate: float = 0.10,
    site_shift: float = 0.3,
    nonlinear: bool = True,
    random_state: Optional[int] = 0,
) -> pd.DataFrame:
    """Generate a synthetic table of ICU-like stays with a binary outcome.

    Returns a DataFrame with identifier columns ``patient_id``, ``stay_id``,
    ``site_id``; predictors ``age``, ``sex``, ``heart_rate``, ``resp_rate``,
    ``mean_bp``, ``temperature``, ``spo2``, ``creatinine``, ``lactate``,
    ``wbc``, ``platelets``, ``gcs``; the outcome ``outcome``; and the
    data-generating probability ``true_risk``.
    """
    if not 0 < prevalence < 1:
        raise ValueError("prevalence must lie in (0, 1).")
    rng = np.random.default_rng(random_state)
    n_patients = max(1, int(round(n_stays / max(stays_per_patient, 1.0))))
    patient_of_stay = np.sort(np.concatenate([np.arange(n_patients), rng.integers(0, n_patients, n_stays - n_patients)]))[:n_stays]
    site_of_patient = rng.integers(0, n_sites, n_patients)
    site = site_of_patient[patient_of_stay]
    site_effect = rng.normal(0, site_shift, n_sites)
    frailty = rng.normal(0, 0.5, n_patients)[patient_of_stay]

    age = np.clip(rng.normal(64 + 4 * site_effect[site], 15, n_stays), 18, 95).round()
    male = rng.random(n_stays) < 0.56
    severity = rng.normal(0, 1, n_stays) + 0.6 * frailty + 0.5 * site_effect[site]
    heart_rate = rng.normal(88 + 9 * severity, 14)
    resp_rate = rng.normal(19 + 2.5 * severity, 4)
    mean_bp = rng.normal(78 - 6 * severity, 11)
    temperature = rng.normal(37.0 + 0.25 * severity, 0.7)
    spo2 = np.clip(rng.normal(96 - 1.6 * np.maximum(severity, 0), 2.5), 70, 100)
    creatinine = np.exp(rng.normal(0.1 + 0.28 * severity, 0.45))
    lactate = np.exp(rng.normal(0.4 + 0.33 * severity, 0.45))
    wbc = np.exp(rng.normal(2.3 + 0.15 * severity, 0.4))
    platelets = np.clip(rng.normal(220 - 25 * severity, 70), 5, None)
    gcs = np.clip(np.round(rng.normal(14 - 1.5 * np.maximum(severity, 0), 2)), 3, 15)

    lp = (
        0.035 * (age - 64)
        + 0.25 * male
        + 0.015 * (heart_rate - 88)
        + 0.04 * (resp_rate - 19)
        - 0.02 * (mean_bp - 78)
        + 0.45 * np.log(creatinine)
        + 0.55 * np.log(lactate)
        - 0.12 * (gcs - 14)
        - 0.002 * (platelets - 220)
        + 0.4 * frailty
        + site_effect[site]
    )
    if nonlinear:
        lp = lp + 0.15 * ((temperature - 37.0) ** 2) + 0.3 * np.maximum(0, 92 - spo2) / 4
    intercept = brentq(lambda a: expit(a + lp).mean() - prevalence, -20, 20)
    true_risk = expit(intercept + lp)
    outcome = (rng.random(n_stays) < true_risk).astype(int)

    frame = pd.DataFrame(
        {
            "patient_id": patient_of_stay + 1,
            "stay_id": np.arange(1, n_stays + 1) + 100000,
            "site_id": site + 1,
            "age": age,
            "sex": np.where(male, "M", "F"),
            "heart_rate": heart_rate.round(1),
            "resp_rate": resp_rate.round(1),
            "mean_bp": mean_bp.round(1),
            "temperature": temperature.round(2),
            "spo2": spo2.round(1),
            "creatinine": creatinine.round(2),
            "lactate": lactate.round(2),
            "wbc": wbc.round(1),
            "platelets": platelets.round(0),
            "gcs": gcs,
        }
    )
    lab_cols = ["creatinine", "lactate", "wbc", "platelets"]
    site_missing = np.clip(missing_rate * (1 + rng.normal(0, 0.8, n_sites)), 0, 0.6)
    for col in lab_cols:
        rate = site_missing[site] * (1.6 if col == "lactate" else 1.0)
        frame.loc[rng.random(n_stays) < rate, col] = np.nan
    frame["outcome"] = outcome
    frame["true_risk"] = true_risk
    return frame


FEATURES = ["age", "sex", "heart_rate", "resp_rate", "mean_bp", "temperature", "spo2", "creatinine", "lactate", "wbc", "platelets", "gcs"]
