"""Flexible (smoothed) calibration curves and calibration-error summaries.

The default smoother is a logistic regression of the outcome on a restricted
cubic spline of ``logit(p)``. It is fast enough to be refitted for hundreds of
bootstrap models on tens of thousands of patients, which a loess smoother is
not. A loess option is available when ``statsmodels`` is installed.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from ._validation import EPS, expit, logit
from .metrics import fit_logistic

_KNOT_QUANTILES = {
    3: (0.10, 0.50, 0.90),
    4: (0.05, 0.35, 0.65, 0.95),
    5: (0.05, 0.275, 0.50, 0.725, 0.95),
    6: (0.05, 0.23, 0.41, 0.59, 0.77, 0.95),
    7: (0.025, 0.1833, 0.3417, 0.50, 0.6583, 0.8167, 0.975),
}

METHODS = ("spline", "lowess", "binned")


def rcs_basis(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Restricted cubic spline basis (linear beyond the outer knots).

    Returns an ``(n, k-1)`` matrix: the linear term followed by ``k-2``
    non-linear terms, scaled for numerical stability.
    """
    x = np.asarray(x, dtype=float)
    k = np.asarray(knots, dtype=float)
    nk = k.size
    scale = (k[-1] - k[0]) ** 2 if k[-1] > k[0] else 1.0
    cols = [x]
    for j in range(nk - 2):
        term = (
            np.maximum(x - k[j], 0) ** 3
            - np.maximum(x - k[-2], 0) ** 3 * (k[-1] - k[j]) / (k[-1] - k[-2])
            + np.maximum(x - k[-1], 0) ** 3 * (k[-2] - k[j]) / (k[-1] - k[-2])
        )
        cols.append(term / scale)
    return np.column_stack(cols)


def _choose_knots(lp: np.ndarray, n_knots: int) -> Optional[np.ndarray]:
    n_knots = int(min(max(n_knots, 3), 7))
    while n_knots >= 3:
        knots = np.unique(np.quantile(lp, _KNOT_QUANTILES[n_knots]))
        if knots.size == n_knots:
            return knots
        n_knots -= 1
    return None


class SplineCalibrator:
    """Fit ``P(y=1 | p)`` with a restricted cubic spline in ``logit(p)``."""

    def __init__(self, n_knots: int = 4, ridge: float = 1e-6):
        self.n_knots = n_knots
        self.ridge = ridge
        self.knots_: Optional[np.ndarray] = None
        self.coef_: Optional[np.ndarray] = None
        self.range_: tuple = (0.0, 1.0)

    def _design(self, lp: np.ndarray) -> np.ndarray:
        if self.knots_ is None:
            return np.column_stack([np.ones(lp.size), lp])
        return np.column_stack([np.ones(lp.size), rcs_basis(lp, self.knots_)])

    def fit(self, y: np.ndarray, p: np.ndarray) -> "SplineCalibrator":
        lp = logit(p)
        self.knots_ = _choose_knots(lp, self.n_knots) if np.unique(lp).size > 10 else None
        self.coef_ = fit_logistic(self._design(lp), y, ridge=self.ridge)
        self.range_ = (float(np.min(p)), float(np.max(p)))
        return self

    def predict(self, p: np.ndarray) -> np.ndarray:
        lp = logit(p)
        return expit(self._design(lp) @ self.coef_)


def _lowess_values(y: np.ndarray, p: np.ndarray, frac: float) -> np.ndarray:
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
    except ImportError as exc:  # pragma: no cover - depends on optional dependency
        raise ImportError(
            "method='lowess' requires statsmodels. Install it with `pip install cpmaudit[lowess]` "
            "or use method='spline'."
        ) from exc
    delta = 0.01 * (np.max(p) - np.min(p))
    fitted = lowess(y, p, frac=frac, it=0, delta=delta, return_sorted=False)
    return np.clip(fitted, 0.0, 1.0)


def calibrated_risk(
    y: np.ndarray,
    p: np.ndarray,
    method: str = "spline",
    n_knots: int = 4,
    frac: float = 0.75,
    n_bins: int = 10,
) -> np.ndarray:
    """Smoothed observed risk at each individual's predicted risk."""
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)
    if method == "spline":
        return SplineCalibrator(n_knots=n_knots).fit(y, p).predict(p)
    if method == "lowess":
        return _lowess_values(y, p, frac)
    if method == "binned":
        edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
        bins = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, max(edges.size - 2, 0))
        means = np.bincount(bins, weights=y) / np.maximum(np.bincount(bins), 1)
        return means[bins]
    raise ValueError(f"method must be one of {METHODS}; got {method!r}.")


def calibration_curve(
    y: np.ndarray,
    p: np.ndarray,
    grid: Optional[np.ndarray] = None,
    method: str = "spline",
    n_knots: int = 4,
    frac: float = 0.75,
    trim: float = 0.005,
) -> Dict[str, np.ndarray]:
    """Smoothed calibration curve evaluated on a grid of predicted risks.

    Grid points outside the ``[trim, 1-trim]`` quantile range of ``p`` are
    returned as NaN so that curves are never extrapolated.
    """
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)
    if grid is None:
        grid = np.linspace(0.01, 0.99, 99)
    grid = np.asarray(grid, dtype=float)
    lo, hi = np.quantile(p, [trim, 1.0 - trim])
    observed = np.full(grid.shape, np.nan)
    inside = (grid >= lo) & (grid <= hi)
    if method == "spline":
        model = SplineCalibrator(n_knots=n_knots).fit(y, p)
        observed[inside] = model.predict(grid[inside])
    elif method == "lowess":
        order = np.argsort(p)
        fitted = _lowess_values(y, p, frac)[order]
        observed[inside] = np.interp(grid[inside], p[order], fitted)
    elif method == "binned":
        fitted = calibrated_risk(y, p, method="binned")
        order = np.argsort(p)
        observed[inside] = np.interp(grid[inside], p[order], fitted[order])
    else:
        raise ValueError(f"method must be one of {METHODS}; got {method!r}.")
    return {"predicted": grid, "observed": observed}


def calibration_error_summary(
    y: np.ndarray, p: np.ndarray, method: str = "spline", n_knots: int = 4
) -> Dict[str, float]:
    """Integrated calibration index (ICI, also called E-avg), E50, E90 and Emax.

    These are the mean, median, 90th percentile and maximum of the absolute
    difference between predicted risk and smoothed observed risk, taken over
    the distribution of predicted risks in the sample.
    """
    y = np.asarray(y, dtype=float)
    if y.min() == y.max():
        return {"ici": float("nan"), "e50": float("nan"), "e90": float("nan"), "emax": float("nan")}
    p = np.asarray(p, dtype=float)
    diff = np.abs(p - calibrated_risk(y, p, method=method, n_knots=n_knots))
    return {
        "ici": float(np.mean(diff)),
        "e50": float(np.median(diff)),
        "e90": float(np.quantile(diff, 0.9)),
        "emax": float(np.max(diff)),
    }
