"""Minimum sample size for developing a risk model for a binary outcome.

Implements the three analytic criteria for binary outcomes proposed by
Riley and colleagues:

1. expected uniform shrinkage of predictor effects of at least ``S`` (0.9);
2. a small absolute difference (``delta``, 0.05) between apparent and
   optimism-adjusted Nagelkerke R-squared;
3. precise estimation of the overall outcome proportion (margin of error
   0.05).

The anticipated Cox-Snell R-squared can be given directly or approximated
from an anticipated C statistic (AUROC) and outcome proportion by simulation.

These criteria are a *starting point*. They assume a model whose complexity
is described by its number of parameters; flexible machine-learning models
usually need more data. Use :func:`cpmaudit.instability_curve` to check
empirically whether individual predictions are stable at a given size.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

from .metrics import fit_logistic


def max_cox_snell_r2(prevalence: float) -> float:
    """Largest achievable Cox-Snell R-squared for a given outcome proportion."""
    _check_prevalence(prevalence)
    phi = prevalence
    ln_null = phi * math.log(phi) + (1 - phi) * math.log(1 - phi)
    return 1.0 - math.exp(2.0 * ln_null)


def cox_snell_r2_from_c(c_statistic: float, prevalence: float, n_sim: int = 1_000_000, random_state: int = 1234) -> float:
    """Approximate the Cox-Snell R-squared implied by a C statistic.

    The linear predictor is simulated as normal with unit variance in
    non-events and events, with a mean difference of ``sqrt(2) * Phi^-1(C)``
    so that the population C statistic equals ``c_statistic``; a logistic
    model is then fitted and its likelihood-ratio R-squared returned.
    """
    if not 0.5 < c_statistic < 1.0:
        raise ValueError("c_statistic must lie strictly between 0.5 and 1.")
    _check_prevalence(prevalence)
    rng = np.random.default_rng(random_state)
    n_events = int(round(prevalence * n_sim))
    mu = math.sqrt(2.0) * norm.ppf(c_statistic)
    lp = np.concatenate([rng.normal(0.0, 1.0, n_sim - n_events), rng.normal(mu, 1.0, n_events)])
    y = np.concatenate([np.zeros(n_sim - n_events), np.ones(n_events)])
    X = np.column_stack([np.ones(n_sim), lp])
    beta = fit_logistic(X, y)
    eta = X @ beta
    ll_model = float(np.sum(y * eta - np.logaddexp(0.0, eta)))
    phi = y.mean()
    ll_null = float(n_sim * (phi * math.log(phi) + (1 - phi) * math.log(1 - phi)))
    lr = 2.0 * (ll_model - ll_null)
    return float(1.0 - math.exp(-lr / n_sim))


@dataclass
class SampleSizeResult:
    criteria: pd.DataFrame
    minimum_n: int
    minimum_events: float
    events_per_parameter: float
    r2_cox_snell: float
    r2_nagelkerke: float
    max_r2_cox_snell: float
    prevalence: float
    n_parameters: int

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        lines = [self.criteria.to_string(index=False), "",
                 f"Minimum sample size: {self.minimum_n} ({self.minimum_events:.1f} events, "
                 f"{self.events_per_parameter:.2f} events per candidate parameter)"]
        return "\n".join(lines)


def minimum_sample_size_binary(
    n_parameters: int,
    prevalence: float,
    r2_cox_snell: Optional[float] = None,
    c_statistic: Optional[float] = None,
    r2_nagelkerke: Optional[float] = None,
    shrinkage: float = 0.9,
    delta_r2: float = 0.05,
    margin_of_error: float = 0.05,
    n_sim: int = 1_000_000,
    random_state: int = 1234,
) -> SampleSizeResult:
    """Minimum development sample size for a binary-outcome risk model.

    Exactly one of ``r2_cox_snell``, ``r2_nagelkerke`` or ``c_statistic`` must
    be supplied as the anticipated model fit. ``n_parameters`` counts all
    candidate predictor parameters (dummy variables, spline terms and
    interactions included), not only the final selected ones.
    """
    if int(n_parameters) < 1:
        raise ValueError("n_parameters must be at least 1.")
    given = [v is not None for v in (r2_cox_snell, c_statistic, r2_nagelkerke)]
    if sum(given) != 1:
        raise ValueError("Specify exactly one of r2_cox_snell, r2_nagelkerke or c_statistic.")
    max_r2 = max_cox_snell_r2(prevalence)
    if c_statistic is not None:
        r2 = cox_snell_r2_from_c(c_statistic, prevalence, n_sim=n_sim, random_state=random_state)
    elif r2_nagelkerke is not None:
        r2 = float(r2_nagelkerke) * max_r2
    else:
        r2 = float(r2_cox_snell)
    if not 0 < r2 < max_r2:
        raise ValueError(f"The anticipated Cox-Snell R-squared must lie in (0, {max_r2:.3f}) for this prevalence.")
    p = int(n_parameters)

    def n_for_shrinkage(s: float) -> float:
        return p / ((s - 1.0) * math.log(1.0 - r2 / s))

    n1 = n_for_shrinkage(shrinkage)
    s2 = r2 / (r2 + delta_r2 * max_r2)
    n2 = n_for_shrinkage(s2)
    n3 = (norm.ppf(0.975) / margin_of_error) ** 2 * prevalence * (1 - prevalence)

    rows = []
    for label, n, s in (
        ("criterion 1: shrinkage", n1, shrinkage),
        ("criterion 2: optimism in R-squared", n2, s2),
        ("criterion 3: overall risk precision", n3, np.nan),
    ):
        n_ceiled = int(math.ceil(n - 1e-9))
        rows.append(
            {
                "criterion": label,
                "sample_size": n_ceiled,
                "events": n_ceiled * prevalence,
                "events_per_parameter": n_ceiled * prevalence / p,
                "shrinkage": s,
            }
        )
    table = pd.DataFrame(rows)
    final = int(table["sample_size"].max())
    final_shrinkage = shrinkage if final == table.loc[0, "sample_size"] else _shrinkage_at(final, p, r2)
    table["shrinkage_at_minimum"] = final_shrinkage
    return SampleSizeResult(
        criteria=table,
        minimum_n=final,
        minimum_events=final * prevalence,
        events_per_parameter=final * prevalence / p,
        r2_cox_snell=r2,
        r2_nagelkerke=r2 / max_r2,
        max_r2_cox_snell=max_r2,
        prevalence=float(prevalence),
        n_parameters=p,
    )


def _shrinkage_at(n: int, p: int, r2: float) -> float:
    """Expected shrinkage factor at sample size ``n`` (Van Houwelingen heuristic)."""
    lr = -n * math.log(1.0 - r2)
    return 1.0 - p / lr if lr > 0 else float("nan")


def _check_prevalence(prevalence: float) -> None:
    if not 0.0 < prevalence < 1.0:
        raise ValueError("prevalence must lie strictly between 0 and 1.")
