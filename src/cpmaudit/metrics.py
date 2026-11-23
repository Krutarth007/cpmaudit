"""Performance measures for risk predictions of a binary outcome.

All functions take observed outcomes ``y`` (0/1) and predicted risks ``p``
(probabilities) and avoid heavy dependencies so they can be called thousands
of times inside bootstrap loops.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np
from scipy.stats import rankdata

from ._validation import EPS, as_binary_outcome, as_probability, logit

__all__ = [
    "roc_auc",
    "brier_score",
    "scaled_brier_score",
    "log_loss",
    "calibration_intercept_slope",
    "observed_expected_ratio",
    "net_benefit",
    "threshold_metrics",
    "performance_summary",
    "fit_logistic",
]


def roc_auc(y: np.ndarray, p: np.ndarray) -> float:
    """Area under the ROC curve (concordance statistic) with tie handling."""
    y = np.asarray(y)
    n1 = int(y.sum())
    n0 = y.size - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    ranks = rankdata(p)
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def brier_score(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((np.asarray(p, dtype=float) - np.asarray(y, dtype=float)) ** 2))


def scaled_brier_score(y: np.ndarray, p: np.ndarray) -> float:
    """Brier score scaled by its value under a prevalence-only model (1 is perfect)."""
    prev = float(np.mean(y))
    denom = prev * (1.0 - prev)
    if denom <= 0:
        return float("nan")
    return 1.0 - brier_score(y, p) / denom


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)
    y = np.asarray(y, dtype=float)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log1p(-p)))


def fit_logistic(
    X: np.ndarray,
    y: np.ndarray,
    offset: Optional[np.ndarray] = None,
    max_iter: int = 100,
    tol: float = 1e-10,
    ridge: float = 0.0,
) -> np.ndarray:
    """Maximum likelihood logistic regression via Newton-Raphson.

    ``X`` must already contain an intercept column if one is wanted. A small
    ``ridge`` penalty can stabilise separated problems. Returns coefficients.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    off = np.zeros(y.shape[0]) if offset is None else np.asarray(offset, dtype=float)
    beta = np.zeros(X.shape[1])
    penalty = ridge * np.eye(X.shape[1])

    def loglik(b: np.ndarray) -> float:
        eta = off + X @ b
        return float(np.sum(y * eta - np.logaddexp(0.0, eta)) - 0.5 * ridge * b @ b)

    current = loglik(beta)
    for _ in range(max_iter):
        eta = off + X @ beta
        mu = 1.0 / (1.0 + np.exp(-np.clip(eta, -35, 35)))
        w = mu * (1.0 - mu)
        grad = X.T @ (y - mu) - ridge * beta
        hess = (X * w[:, None]).T @ X + penalty
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hess, grad, rcond=None)[0]
        t = 1.0
        while t > 1e-6:
            candidate = beta + t * step
            new = loglik(candidate)
            if new >= current - 1e-12:
                break
            t *= 0.5
        beta = candidate
        if abs(new - current) < tol * (abs(current) + 1.0):
            current = new
            break
        current = new
    return beta


def calibration_intercept_slope(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    """Calibration-in-the-large (intercept with the slope fixed at 1) and
    calibration slope from logistic recalibration on ``logit(p)``."""
    y = np.asarray(y, dtype=float)
    lp = logit(p)
    if y.min() == y.max():
        return {"calibration_intercept": float("nan"), "calibration_slope": float("nan")}
    ones = np.ones((y.size, 1))
    citl = fit_logistic(ones, y, offset=lp)[0]
    slope = fit_logistic(np.column_stack([ones, lp]), y)[1]
    return {"calibration_intercept": float(citl), "calibration_slope": float(slope)}


def observed_expected_ratio(y: np.ndarray, p: np.ndarray) -> float:
    expected = float(np.sum(p))
    return float(np.sum(y) / expected) if expected > 0 else float("nan")


def net_benefit(y: np.ndarray, p: np.ndarray, thresholds: Iterable[float]) -> np.ndarray:
    """Net benefit of treating patients with ``p >= t`` for each threshold ``t``."""
    y = np.asarray(y, dtype=bool)
    p = np.asarray(p, dtype=float)
    n = y.size
    out = []
    for t in thresholds:
        flag = p >= t
        tp = np.sum(flag & y)
        fp = np.sum(flag & ~y)
        out.append(tp / n - fp / n * (t / (1.0 - t)))
    return np.asarray(out, dtype=float)


def net_benefit_treat_all(y: np.ndarray, thresholds: Iterable[float]) -> np.ndarray:
    prev = float(np.mean(y))
    return np.asarray([prev - (1.0 - prev) * t / (1.0 - t) for t in thresholds], dtype=float)


def threshold_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> Dict[str, float]:
    """Classification measures when patients with ``p >= threshold`` are flagged."""
    y = np.asarray(y, dtype=bool)
    flag = np.asarray(p) >= threshold
    tp = float(np.sum(flag & y))
    fp = float(np.sum(flag & ~y))
    fn = float(np.sum(~flag & y))
    tn = float(np.sum(~flag & ~y))

    def ratio(a: float, b: float) -> float:
        return a / b if b > 0 else float("nan")

    return {
        "threshold": float(threshold),
        "flagged_fraction": float(np.mean(flag)),
        "sensitivity": ratio(tp, tp + fn),
        "specificity": ratio(tn, tn + fp),
        "ppv": ratio(tp, tp + fp),
        "npv": ratio(tn, tn + fn),
        "net_benefit": float(net_benefit(y, p, [threshold])[0]),
    }


def performance_summary(
    y,
    p,
    thresholds: Iterable[float] = (),
    calibration: bool = True,
    smooth_calibration: bool = True,
) -> Dict[str, float]:
    """A compact set of discrimination, calibration and overall measures.

    The set follows current recommendations for evaluating risk models:
    discrimination (AUROC), calibration (intercept, slope, O/E and smoothed
    calibration error), overall accuracy (Brier, scaled Brier, log loss) and
    clinical utility (net benefit at the requested thresholds).
    """
    y = as_binary_outcome(y)
    p = as_probability(p)
    out: Dict[str, float] = {
        "n": float(y.size),
        "events": float(y.sum()),
        "prevalence": float(y.mean()),
        "mean_risk": float(p.mean()),
        "auroc": roc_auc(y, p),
        "brier": brier_score(y, p),
        "scaled_brier": scaled_brier_score(y, p),
        "log_loss": log_loss(y, p),
        "oe_ratio": observed_expected_ratio(y, p),
    }
    if calibration:
        out.update(calibration_intercept_slope(y, p))
    if smooth_calibration:
        from .calibration import calibration_error_summary

        out.update(calibration_error_summary(y, p))
    for t in thresholds:
        out[f"net_benefit@{t:g}"] = float(net_benefit(y, p, [t])[0])
    return out
