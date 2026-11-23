"""Predictive multiplicity: do equally good models disagree about individuals?

A *Rashomon set* is the set of candidate models whose performance on held-out
data lies within ``epsilon`` of the best candidate. If those near-equivalent
models assign conflicting decisions to the same person, the decision for that
person is arbitrary with respect to model choice.

Measures (computed on decisions at each threshold):

* ambiguity   - share of individuals for whom at least one Rashomon model
  disagrees with the reference (best) model;
* discrepancy - largest share of individuals whose decision would change by
  swapping the reference model for one other Rashomon model;
* pairwise disagreement - average share of model pairs that disagree;

and on risks:

* viable prediction range - per-individual range (max - min) of risks across
  the Rashomon set.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from itertools import product
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from ._validation import as_binary_outcome, as_thresholds, check_features, check_same_length
from .metrics import brier_score, log_loss, roc_auc
from .models import ModelBuilder, fit_quietly

_METRICS = {
    "auroc": (roc_auc, True),
    "log_loss": (log_loss, False),
    "brier": (brier_score, False),
}


def rashomon_set(scores: Sequence[float], epsilon: float, higher_is_better: bool = True) -> np.ndarray:
    """Boolean mask of candidates within ``epsilon`` of the best score."""
    s = np.asarray(scores, dtype=float)
    if np.isnan(s).all():
        raise ValueError("All candidate scores are missing.")
    best = np.nanmax(s) if higher_is_better else np.nanmin(s)
    return (s >= best - epsilon) if higher_is_better else (s <= best + epsilon)


def multiplicity_measures(
    P: np.ndarray, thresholds: Iterable[float], reference: int = 0, vpr_tolerance: float = 0.05
) -> Dict[str, float]:
    """Multiplicity measures for an ``n_models x n_individuals`` risk matrix."""
    P = np.asarray(P, dtype=float)
    m = P.shape[0]
    vpr = P.max(axis=0) - P.min(axis=0)
    out: Dict[str, float] = {
        "models": float(m),
        "vpr_mean": float(vpr.mean()),
        "vpr_median": float(np.median(vpr)),
        "vpr_p90": float(np.quantile(vpr, 0.9)),
        "share_vpr_above_tolerance": float(np.mean(vpr > vpr_tolerance)),
    }
    for t in thresholds:
        D = P >= t
        ref = D[reference]
        differs = D != ref[None, :]
        k = D.sum(axis=0)
        pairs = m * (m - 1) / 2.0
        out[f"ambiguity@{t:g}"] = float(np.mean(differs.any(axis=0)))
        out[f"discrepancy@{t:g}"] = float(differs.mean(axis=1).max()) if m > 1 else 0.0
        out[f"pairwise_disagreement@{t:g}"] = float(np.mean(k * (m - k) / pairs)) if m > 1 else 0.0
    return out


def seed_variants(model: Any, seeds: Union[int, Sequence[int]] = 20, prefix: str = "seed") -> List[Tuple[str, Any, int]]:
    """Candidate list that refits the same specification with different seeds
    (meaningful for stochastic learners such as forests and boosting)."""
    seeds = list(range(seeds)) if isinstance(seeds, int) else list(seeds)
    return [(f"{prefix}={s}", model, int(s)) for s in seeds]


def param_grid_variants(model: Any, param_grid: Dict[str, Sequence[Any]], seed: int = 0) -> List[Tuple[str, Any, int]]:
    """Candidate list from a hyper-parameter grid of a scikit-learn estimator."""
    from sklearn.base import clone

    keys = sorted(param_grid)
    out = []
    for values in product(*(param_grid[k] for k in keys)):
        params = dict(zip(keys, values))
        est = clone(model).set_params(**params)
        name = ", ".join(f"{k}={v}" for k, v in params.items())
        out.append((name, est, seed))
    return out


def _normalise_candidates(candidates: Any) -> List[Tuple[str, Any, int]]:
    if isinstance(candidates, dict):
        return [(str(k), v, 0) for k, v in candidates.items()]
    out = []
    for i, c in enumerate(candidates):
        if isinstance(c, tuple):
            if len(c) == 3:
                out.append((str(c[0]), c[1], int(c[2])))
            elif len(c) == 2:
                out.append((str(c[0]), c[1], 0))
            else:
                raise ValueError("Candidate tuples must be (name, model) or (name, model, seed).")
        else:
            name = c if isinstance(c, str) else f"model_{i}"
            out.append((name, c, 0))
    if len(out) < 2:
        raise ValueError("At least two candidate models are needed to measure multiplicity.")
    return out


class MultiplicityAudit:
    """Fit candidate models, find the Rashomon set on held-out data and measure
    how much its members disagree about individuals.

    Parameters
    ----------
    candidates:
        A list of preset names / estimators / callables, a list of
        ``(name, model)`` or ``(name, model, seed)`` tuples, or a dict
        ``{name: model}``. See :func:`seed_variants` and :func:`param_grid_variants`.
    epsilon:
        Tolerance defining near-equivalence on the selection metric.
    metric:
        ``"auroc"`` (default), ``"log_loss"`` or ``"brier"``.
    thresholds:
        Decision thresholds for ambiguity and discrepancy.
    """

    def __init__(
        self,
        candidates: Any,
        epsilon: float = 0.01,
        metric: str = "auroc",
        thresholds: Iterable[float] = (0.1,),
        vpr_tolerance: float = 0.05,
        n_jobs: int = 1,
    ):
        if metric not in _METRICS:
            raise ValueError(f"metric must be one of {tuple(_METRICS)}.")
        self.candidates = _normalise_candidates(candidates)
        self.epsilon = float(epsilon)
        self.metric = metric
        self.thresholds = as_thresholds(thresholds)
        self.vpr_tolerance = float(vpr_tolerance)
        self.n_jobs = n_jobs

    def run(self, X_train: Any, y_train: Any, X_eval: Any, y_eval: Any, X_target: Any = None) -> "MultiplicityResult":
        X_train = check_features(X_train, "X_train")
        X_eval = check_features(X_eval, "X_eval")
        y_train = as_binary_outcome(y_train, "y_train")
        y_eval = as_binary_outcome(y_eval, "y_eval")
        check_same_length(X_train, y_train, names=("X_train", "y_train"))
        check_same_length(X_eval, y_eval, names=("X_eval", "y_eval"))
        target = X_eval if X_target is None else check_features(X_target, "X_target")
        scorer, higher = _METRICS[self.metric]

        def fit_one(name: str, spec: Any, seed: int):
            builder = ModelBuilder(spec)
            try:
                model, _ = fit_quietly(builder, X_train, y_train, random_state=seed)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    p_eval = builder.predict_risk(model, X_eval)
                    p_target = p_eval if X_target is None else builder.predict_risk(model, target)
                return name, scorer(y_eval, p_eval), p_target, None
            except Exception as exc:  # noqa: BLE001
                return name, np.nan, None, f"{type(exc).__name__}: {exc}"

        if self.n_jobs in (None, 1):
            fitted = [fit_one(*c) for c in self.candidates]
        else:
            from joblib import Parallel, delayed

            fitted = Parallel(n_jobs=self.n_jobs)(delayed(fit_one)(*c) for c in self.candidates)

        ok = [f for f in fitted if f[3] is None]
        failures = [{"candidate": f[0], "error": f[3]} for f in fitted if f[3] is not None]
        if len(ok) < 2:
            raise RuntimeError(f"Fewer than two candidates could be fitted: {failures[:3]}")
        names = [f[0] for f in ok]
        scores = np.array([f[1] for f in ok], dtype=float)
        P = np.vstack([f[2] for f in ok])
        mask = rashomon_set(scores, self.epsilon, higher)
        best = int(np.nanargmax(scores) if higher else np.nanargmin(scores))
        members = np.flatnonzero(mask)
        reference = int(np.flatnonzero(members == best)[0])
        table = pd.DataFrame({"candidate": names, self.metric: scores, "in_rashomon_set": mask, "reference": np.arange(len(names)) == best})
        return MultiplicityResult(
            candidates=table,
            P=P[mask],
            reference_index=reference,
            thresholds=self.thresholds,
            metric=self.metric,
            epsilon=self.epsilon,
            vpr_tolerance=self.vpr_tolerance,
            failures=failures,
        )


@dataclass
class MultiplicityResult:
    candidates: pd.DataFrame
    P: np.ndarray
    reference_index: int
    thresholds: tuple
    metric: str
    epsilon: float
    vpr_tolerance: float = 0.05
    failures: List[Dict[str, Any]] = field(default_factory=list)

    def summary(self) -> pd.DataFrame:
        measures = multiplicity_measures(self.P, self.thresholds, self.reference_index, self.vpr_tolerance)
        return pd.DataFrame({"measure": list(measures), "value": list(measures.values())})

    def individual(self) -> pd.DataFrame:
        P = self.P
        ref = P[self.reference_index]
        out = pd.DataFrame({"risk_reference": ref, "risk_min": P.min(axis=0), "risk_max": P.max(axis=0)})
        out["viable_prediction_range"] = out["risk_max"] - out["risk_min"]
        for t in self.thresholds:
            D = P >= t
            out[f"decision_reference@{t:g}"] = D[self.reference_index].astype(np.int8)
            out[f"ambiguous@{t:g}"] = (D != D[self.reference_index][None, :]).any(axis=0)
            out[f"share_models_above@{t:g}"] = D.mean(axis=0)
        return out


def multiplicity_from_predictions(
    P: np.ndarray,
    scores: Optional[Sequence[float]] = None,
    epsilon: float = 0.01,
    higher_is_better: bool = True,
    thresholds: Iterable[float] = (0.1,),
    vpr_tolerance: float = 0.05,
) -> MultiplicityResult:
    """Measure multiplicity from an existing matrix of model predictions.

    ``P`` is ``n_models x n_individuals``. If ``scores`` are given, only models
    within ``epsilon`` of the best are kept and the best model is the reference.
    """
    P = np.asarray(P, dtype=float)
    if scores is None:
        mask = np.ones(P.shape[0], dtype=bool)
        best = 0
        scores_arr = np.full(P.shape[0], np.nan)
    else:
        scores_arr = np.asarray(scores, dtype=float)
        mask = rashomon_set(scores_arr, epsilon, higher_is_better)
        best = int(np.nanargmax(scores_arr) if higher_is_better else np.nanargmin(scores_arr))
    members = np.flatnonzero(mask)
    reference = int(np.flatnonzero(members == best)[0]) if best in members else 0
    table = pd.DataFrame(
        {"candidate": [f"model_{i}" for i in range(P.shape[0])], "score": scores_arr, "in_rashomon_set": mask,
         "reference": np.arange(P.shape[0]) == best}
    )
    return MultiplicityResult(table, P[mask], reference, as_thresholds(thresholds), "score", float(epsilon), vpr_tolerance)
