"""Bootstrap audit of individual-level prediction instability.

The audit answers one question: *if the same model-building process had been
run on a different sample of the same size from the same population, how
different would each individual's predicted risk, and the decision based on
it, have been?*

Procedure
---------
1. Fit the model on the full development data (the "original" model) and
   record each individual's predicted risk.
2. Draw ``B`` bootstrap replicates of the development data (row, stratified or
   cluster bootstrap). In each replicate repeat the *entire* model-building
   process and predict risks for the original individuals (and, optionally,
   for an external population).
3. Summarise the spread of the ``B`` risks around the original risk for every
   individual, the frequency with which each individual's decision changes at
   clinically relevant thresholds, and how calibration, net benefit and
   overall performance vary across replicates.

The same replicates also give Harrell's bootstrap optimism correction of
apparent performance at no extra cost.
"""

from __future__ import annotations

import datetime as _dt
import json
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from ._validation import (
    as_binary_outcome,
    as_label_array,
    as_thresholds,
    check_features,
    check_same_length,
    n_rows,
    take_rows,
)
from ._version import __version__
from .calibration import calibration_curve, calibration_error_summary
from .metrics import (
    brier_score,
    calibration_intercept_slope,
    log_loss,
    net_benefit,
    net_benefit_treat_all,
    observed_expected_ratio,
    roc_auc,
    scaled_brier_score,
)
from .models import ModelBuilder, fit_quietly
from .resampling import BootstrapSampler, out_of_bag_mask

POPULATIONS = ("development", "external")

LOWER_IS_BETTER = {"brier", "log_loss", "ici", "e50", "e90", "emax"}


def core_metrics(y: np.ndarray, p: np.ndarray, thresholds: Sequence[float] = (), smooth: bool = True) -> Dict[str, float]:
    """Metrics computed for every replicate (kept small for speed)."""
    out = {
        "auroc": roc_auc(y, p),
        "brier": brier_score(y, p),
        "scaled_brier": scaled_brier_score(y, p),
        "log_loss": log_loss(y, p),
        "oe_ratio": observed_expected_ratio(y, p),
        "mean_risk": float(np.mean(p)),
    }
    out.update(calibration_intercept_slope(y, p))
    if smooth:
        out.update(calibration_error_summary(y, p))
    for t in thresholds:
        out[f"net_benefit@{t:g}"] = float(net_benefit(y, p, [t])[0])
    return out


@dataclass
class _Replicate:
    index: int
    seed: int
    ok: bool
    error: Optional[str] = None
    risk_dev: Optional[np.ndarray] = None
    risk_ext: Optional[np.ndarray] = None
    apparent: Optional[Dict[str, float]] = None
    test: Optional[Dict[str, float]] = None
    external: Optional[Dict[str, float]] = None
    oob: Optional[Dict[str, float]] = None
    convergence_warnings: int = 0
    n_rows: int = 0
    n_events: int = 0
    seconds: float = 0.0
    model: Any = None


def _run_replicate(
    index: int,
    seed: int,
    builder: ModelBuilder,
    X: Any,
    y: np.ndarray,
    sampler: BootstrapSampler,
    X_ext: Any,
    y_ext: Optional[np.ndarray],
    thresholds: Sequence[float],
    optimism: bool,
    smooth: bool,
    keep_model: bool,
) -> _Replicate:
    start = time.perf_counter()
    rng = np.random.default_rng(seed)
    try:
        idx = sampler.draw(y.size, rng, y)
        Xb, yb = take_rows(X, idx), y[idx]
        model, n_conv = fit_quietly(builder, Xb, yb, random_state=seed % (2**31 - 1))
        risk_dev = builder.predict_risk(model, X)
        rep = _Replicate(
            index=index,
            seed=seed,
            ok=True,
            risk_dev=risk_dev.astype(np.float32),
            convergence_warnings=n_conv,
            n_rows=int(idx.size),
            n_events=int(yb.sum()),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            rep.test = core_metrics(y, risk_dev, thresholds, smooth)
            if optimism:
                rep.apparent = core_metrics(yb, builder.predict_risk(model, Xb), thresholds, smooth)
                oob = out_of_bag_mask(y.size, idx)
                if oob.sum() >= 10 and np.unique(y[oob]).size == 2:
                    rep.oob = core_metrics(y[oob], risk_dev[oob], thresholds, smooth)
            if X_ext is not None:
                risk_ext = builder.predict_risk(model, X_ext)
                rep.risk_ext = risk_ext.astype(np.float32)
                if y_ext is not None:
                    rep.external = core_metrics(y_ext, risk_ext, thresholds, smooth)
        if keep_model:
            rep.model = model
    except Exception as exc:  # noqa: BLE001 - replicate failures are recorded, not fatal
        rep = _Replicate(index=index, seed=seed, ok=False, error=f"{type(exc).__name__}: {exc}")
    rep.seconds = time.perf_counter() - start
    return rep


class StabilityAudit:
    """Audit the instability of individual risk predictions by bootstrapping
    the complete model-building process.

    Parameters
    ----------
    model:
        Preset name, scikit-learn compatible estimator/pipeline, or callable
        ``build(X, y, random_state=...)`` (see :mod:`cpmaudit.models`).
    n_bootstrap:
        Number of bootstrap replicates. 200 is a practical minimum; 500-1000
        gives smoother individual intervals.
    bootstrap:
        ``"simple"``, ``"stratified"`` (keeps the number of events fixed) or
        ``"cluster"`` (resamples whole patients; requires ``groups`` in
        :meth:`run`). Use ``"cluster"`` whenever patients contribute more than
        one row.
    thresholds:
        Risk thresholds at which decisions are made (for example ``0.1`` if
        patients above 10% risk receive an intervention).
    interval_level:
        Coverage of the per-individual percentile interval of bootstrap risks.
    optimism:
        Also compute Harrell's bootstrap optimism-corrected performance.
    keep_models:
        Store fitted replicate models so that :meth:`StabilityResult.predict_interval`
        can return uncertainty intervals for *new* individuals. Increases memory use.
    n_jobs:
        Parallel workers (``-1`` uses all cores). Keep model-level parallelism at
        one thread when using several workers.
    random_state:
        Seed for full reproducibility.
    predict_fn:
        Optional ``predict_fn(model, X)`` returning risks, for models without
        ``predict_proba``.
    max_failure_fraction:
        Abort if more than this fraction of replicates fail to fit.
    smooth_calibration:
        Compute smoothed calibration errors (ICI, E50, E90, Emax) per replicate.
    verbose:
        0 silent, 1 progress messages.
    """

    def __init__(
        self,
        model: Any = "logistic",
        n_bootstrap: int = 200,
        bootstrap: str = "simple",
        thresholds: Iterable[float] = (0.1,),
        interval_level: float = 0.95,
        optimism: bool = True,
        keep_models: bool = False,
        n_jobs: int = 1,
        random_state: Optional[int] = None,
        predict_fn: Any = None,
        max_failure_fraction: float = 0.05,
        smooth_calibration: bool = True,
        verbose: int = 0,
    ):
        if int(n_bootstrap) < 2:
            raise ValueError("n_bootstrap must be at least 2.")
        if not 0.5 <= float(interval_level) < 1.0:
            raise ValueError("interval_level must be in [0.5, 1).")
        self.builder = ModelBuilder(model, predict_fn=predict_fn)
        self.n_bootstrap = int(n_bootstrap)
        self.bootstrap = bootstrap
        self.thresholds = as_thresholds(thresholds)
        self.interval_level = float(interval_level)
        self.optimism = bool(optimism)
        self.keep_models = bool(keep_models)
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.max_failure_fraction = float(max_failure_fraction)
        self.smooth_calibration = bool(smooth_calibration)
        self.verbose = int(verbose)

    def run(
        self,
        X: Any,
        y: Any,
        groups: Any = None,
        subgroups: Any = None,
        ids: Any = None,
        X_external: Any = None,
        y_external: Any = None,
        subgroups_external: Any = None,
        ids_external: Any = None,
    ) -> "StabilityResult":
        """Run the audit.

        Parameters
        ----------
        X, y:
            Development features and binary outcome.
        groups:
            Cluster labels (for example patient identifiers) used by the
            cluster bootstrap.
        subgroups:
            Optional labels (sex, age band, site, ...) for subgroup-level
            stability summaries.
        ids:
            Optional row identifiers carried into per-individual outputs.
        X_external, y_external:
            Optional population on which every replicate model is also
            evaluated (for example another hospital or database). Outcomes are
            optional: without them only prediction instability is measured.
        subgroups_external, ids_external:
            As above, for the external population.
        """
        started = time.perf_counter()
        X = check_features(X, "X")
        y = as_binary_outcome(y, "y")
        check_same_length(X, y, names=("X", "y"))
        n = y.size
        if y.sum() == 0 or y.sum() == n:
            raise ValueError("The outcome must contain both events and non-events.")
        groups_arr = as_label_array(groups, n, "groups") if groups is not None else None
        sub_arr = as_label_array(subgroups, n, "subgroups") if subgroups is not None else None
        ids_arr = as_label_array(ids, n, "ids") if ids is not None else None

        X_ext = y_ext = sub_ext = ids_ext = None
        if X_external is not None:
            X_ext = check_features(X_external, "X_external")
            m = n_rows(X_ext)
            if y_external is not None:
                y_ext = as_binary_outcome(y_external, "y_external")
                check_same_length(X_ext, y_ext, names=("X_external", "y_external"))
            sub_ext = as_label_array(subgroups_external, m, "subgroups_external") if subgroups_external is not None else None
            ids_ext = as_label_array(ids_external, m, "ids_external") if ids_external is not None else None
        elif y_external is not None:
            raise ValueError("y_external was given without X_external.")

        if self.bootstrap == "cluster" and groups_arr is None:
            raise ValueError("bootstrap='cluster' requires groups (for example patient identifiers).")
        sampler = BootstrapSampler(
            scheme=self.bootstrap,
            groups=groups_arr if self.bootstrap == "cluster" else None,
            strata=y if self.bootstrap == "stratified" else None,
        )

        seq = np.random.SeedSequence(self.random_state)
        seeds = [int(s.generate_state(1)[0]) for s in seq.spawn(self.n_bootstrap + 1)]

        if self.verbose:
            print(f"[cpmaudit] fitting original model on {n} rows ({int(y.sum())} events)")
        original_model, orig_conv = fit_quietly(self.builder, X, y, random_state=seeds[0] % (2**31 - 1))
        p_orig = self.builder.predict_risk(original_model, X)
        p_orig_ext = self.builder.predict_risk(original_model, X_ext) if X_ext is not None else None

        args = (X, y, sampler, X_ext, y_ext, self.thresholds, self.optimism, self.smooth_calibration, self.keep_models)
        replicates = self._execute(seeds[1:], args)

        ok = [r for r in replicates if r.ok]
        failed = [r for r in replicates if not r.ok]
        if len(failed) > self.max_failure_fraction * self.n_bootstrap:
            first = "; ".join(sorted({r.error for r in failed})[:3])
            raise RuntimeError(
                f"{len(failed)} of {self.n_bootstrap} bootstrap fits failed (limit "
                f"{self.max_failure_fraction:.0%}). First errors: {first}"
            )
        if len(ok) < 2:
            raise RuntimeError("Fewer than two bootstrap replicates succeeded.")

        P = np.vstack([r.risk_dev for r in ok])
        P_ext = np.vstack([r.risk_ext for r in ok]) if X_ext is not None else None

        rows = []
        for r in ok:
            for sample, metrics in (
                ("bootstrap_apparent", r.apparent),
                ("bootstrap_test", r.test),
                ("bootstrap_oob", r.oob),
                ("external", r.external),
            ):
                if metrics is not None:
                    rows.append({"replicate": r.index, "seed": r.seed, "sample": sample, **metrics})
        replicate_metrics = pd.DataFrame(rows)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            apparent = core_metrics(y, p_orig, self.thresholds, self.smooth_calibration)
            external_original = (
                core_metrics(y_ext, p_orig_ext, self.thresholds, self.smooth_calibration) if y_ext is not None else None
            )

        import sklearn

        config = {
            "package_version": __version__,
            "sklearn_version": sklearn.__version__,
            "numpy_version": np.__version__,
            "created": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            "model": self.builder.description,
            "bootstrap": self.bootstrap,
            "n_bootstrap_requested": self.n_bootstrap,
            "n_bootstrap_effective": len(ok),
            "n_failed": len(failed),
            "random_state": self.random_state,
            "thresholds": list(self.thresholds),
            "interval_level": self.interval_level,
            "n_development": int(n),
            "events_development": int(y.sum()),
            "n_clusters": sampler.n_clusters,
            "n_external": int(n_rows(X_ext)) if X_ext is not None else None,
            "events_external": int(y_ext.sum()) if y_ext is not None else None,
            "n_features": int(X.shape[1]) if hasattr(X, "shape") and len(X.shape) == 2 else None,
            "convergence_warnings": int(orig_conv + sum(r.convergence_warnings for r in ok)),
            "runtime_seconds": round(time.perf_counter() - started, 2),
        }
        if self.verbose:
            print(f"[cpmaudit] done: {len(ok)} replicates in {config['runtime_seconds']} s")

        return StabilityResult(
            y=y,
            p_original=p_orig,
            P=P,
            thresholds=self.thresholds,
            interval_level=self.interval_level,
            subgroups=sub_arr,
            ids=ids_arr,
            y_external=y_ext,
            p_original_external=p_orig_ext,
            P_external=P_ext,
            subgroups_external=sub_ext,
            ids_external=ids_ext,
            performance_apparent=apparent,
            performance_external_original=external_original,
            replicate_metrics=replicate_metrics,
            failures=[{"replicate": r.index, "error": r.error} for r in failed],
            config=config,
            original_model=original_model,
            models=[r.model for r in ok] if self.keep_models else None,
            builder=self.builder,
        )

    def _execute(self, seeds: List[int], args: tuple) -> List[_Replicate]:
        if self.n_jobs in (None, 1):
            out = []
            step = max(1, len(seeds) // 10)
            for i, s in enumerate(seeds):
                out.append(_run_replicate(i, s, self.builder, *args))
                if self.verbose and (i + 1) % step == 0:
                    print(f"[cpmaudit] {i + 1}/{len(seeds)} replicates")
            return out
        from joblib import Parallel, delayed

        try:
            from joblib import parallel_config

            ctx = parallel_config(backend="loky", inner_max_num_threads=1)
        except ImportError:  # pragma: no cover - joblib < 1.3
            from joblib import parallel_backend

            ctx = parallel_backend("loky", inner_max_num_threads=1)
        with ctx:
            return list(
                Parallel(n_jobs=self.n_jobs, verbose=10 if self.verbose else 0)(
                    delayed(_run_replicate)(i, s, self.builder, *args) for i, s in enumerate(seeds)
                )
            )


def stability_audit(model: Any, X: Any, y: Any, **kwargs: Any) -> "StabilityResult":
    """Functional shortcut: ``StabilityAudit(model, **settings).run(X, y, **data)``.

    Keyword arguments accepted by :meth:`StabilityAudit.run` (``groups``,
    ``subgroups``, ``ids``, ``X_external``, ``y_external``, ...) are routed to
    ``run``; everything else configures the audit.
    """
    run_keys = {"groups", "subgroups", "ids", "X_external", "y_external", "subgroups_external", "ids_external"}
    run_kwargs = {k: kwargs.pop(k) for k in list(kwargs) if k in run_keys}
    return StabilityAudit(model, **kwargs).run(X, y, **run_kwargs)


@dataclass
class StabilityResult:
    """Container for audit outputs with summary, export and plotting helpers.

    The main arrays are ``p_original`` (risk from the model fitted on all the
    development data) and ``P`` (an ``n_replicates x n_individuals`` matrix of
    risks from the bootstrap models), plus their external counterparts.
    """

    y: np.ndarray
    p_original: np.ndarray
    P: np.ndarray
    thresholds: tuple
    interval_level: float
    subgroups: Optional[np.ndarray] = None
    ids: Optional[np.ndarray] = None
    y_external: Optional[np.ndarray] = None
    p_original_external: Optional[np.ndarray] = None
    P_external: Optional[np.ndarray] = None
    subgroups_external: Optional[np.ndarray] = None
    ids_external: Optional[np.ndarray] = None
    performance_apparent: Dict[str, float] = field(default_factory=dict)
    performance_external_original: Optional[Dict[str, float]] = None
    replicate_metrics: pd.DataFrame = field(default_factory=pd.DataFrame)
    failures: List[Dict[str, Any]] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)
    original_model: Any = None
    models: Optional[List[Any]] = None
    builder: Optional[ModelBuilder] = None
    flip_tolerance: float = 0.1
    wide_interval: float = 0.1
    _cache: Dict[str, Any] = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------------ access
    @property
    def n_replicates(self) -> int:
        return int(self.P.shape[0])

    @property
    def has_external(self) -> bool:
        return self.P_external is not None

    def _arrays(self, population: str):
        if population not in POPULATIONS:
            raise ValueError(f"population must be one of {POPULATIONS}.")
        if population == "development":
            return self.y, self.p_original, self.P, self.subgroups, self.ids
        if not self.has_external:
            raise ValueError("No external population was supplied to the audit.")
        return self.y_external, self.p_original_external, self.P_external, self.subgroups_external, self.ids_external

    # ------------------------------------------------------- individual level
    def individual(self, population: str = "development") -> pd.DataFrame:
        """One row per individual: original risk, bootstrap spread and, for each
        threshold, the original decision and the classification instability
        index (share of bootstrap models whose decision differs)."""
        key = f"individual:{population}"
        if key in self._cache:
            return self._cache[key].copy()
        y, p, P, sub, ids = self._arrays(population)
        alpha = (1.0 - self.interval_level) / 2.0
        n = p.size
        cols: Dict[str, np.ndarray] = {
            name: np.empty(n)
            for name in ("risk_bootstrap_mean", "risk_bootstrap_median", "risk_bootstrap_sd", "risk_lower", "risk_upper", "mape", "max_abs_change")
        }
        for t in self.thresholds:
            cols[f"share_above@{t:g}"] = np.empty(n)
            cols[f"cii@{t:g}"] = np.empty(n)
        block = max(1, int(4_000_000 // max(P.shape[0], 1)))  # bounds peak memory on large populations
        for start in range(0, n, block):
            sl = slice(start, min(n, start + block))
            Pb = P[:, sl].astype(np.float64)
            pb = p[sl]
            lower, median, upper = np.quantile(Pb, [alpha, 0.5, 1.0 - alpha], axis=0)
            absdiff = np.abs(Pb - pb[None, :])
            cols["risk_bootstrap_mean"][sl] = Pb.mean(axis=0)
            cols["risk_bootstrap_median"][sl] = median
            cols["risk_bootstrap_sd"][sl] = Pb.std(axis=0, ddof=1)
            cols["risk_lower"][sl] = lower
            cols["risk_upper"][sl] = upper
            cols["mape"][sl] = absdiff.mean(axis=0)
            cols["max_abs_change"][sl] = absdiff.max(axis=0)
            for t in self.thresholds:
                above = Pb >= t
                cols[f"share_above@{t:g}"][sl] = above.mean(axis=0)
                cols[f"cii@{t:g}"][sl] = (above != (pb >= t)[None, :]).mean(axis=0)
        frame = pd.DataFrame({"risk": p})
        for name in ("risk_bootstrap_mean", "risk_bootstrap_median", "risk_bootstrap_sd", "risk_lower", "risk_upper"):
            frame[name] = cols[name]
        frame["interval_width"] = frame["risk_upper"] - frame["risk_lower"]
        frame["mape"] = cols["mape"]
        frame["max_abs_change"] = cols["max_abs_change"]
        for t in self.thresholds:
            frame[f"decision@{t:g}"] = (p >= t).astype(np.int8)
            frame[f"share_above@{t:g}"] = cols[f"share_above@{t:g}"]
            frame[f"cii@{t:g}"] = cols[f"cii@{t:g}"]
        if y is not None:
            frame.insert(0, "outcome", y)
        if sub is not None:
            frame.insert(0, "subgroup", sub)
        if ids is not None:
            frame.insert(0, "id", ids)
        self._cache[key] = frame
        return frame.copy()

    def instability_index(self, population: str = "development") -> float:
        """Mean absolute prediction error between bootstrap and original risks,
        averaged over individuals."""
        return float(self.individual(population)["mape"].mean())

    def flag_unstable(
        self,
        population: str = "development",
        max_interval_width: Optional[float] = None,
        max_cii: Optional[float] = None,
    ) -> np.ndarray:
        """Boolean mask of individuals whose prediction should not be acted on
        without review: interval wider than ``max_interval_width`` or, at any
        threshold, a CII above ``max_cii``."""
        ind = self.individual(population)
        width = self.wide_interval if max_interval_width is None else max_interval_width
        tol = self.flip_tolerance if max_cii is None else max_cii
        mask = ind["interval_width"].to_numpy() > width
        for t in self.thresholds:
            mask |= ind[f"cii@{t:g}"].to_numpy() > tol
        return mask

    # ------------------------------------------------------------ summaries
    def _population_summary(self, population: str) -> List[Dict[str, Any]]:
        y, p, P, _, _ = self._arrays(population)
        ind = self.individual(population)
        rep_means = P.mean(axis=1)
        alpha = (1.0 - self.interval_level) / 2.0
        rows = [
            ("n", float(p.size), "individuals"),
            ("events", float(y.sum()) if y is not None else np.nan, "observed events"),
            ("replicates", float(self.n_replicates), "successful bootstrap models"),
            ("mean_risk_original", float(p.mean()), "level 1: mean predicted risk, original model"),
            ("mean_risk_replicates_mean", float(rep_means.mean()), "level 1: mean predicted risk averaged over replicates"),
            ("mean_risk_replicates_lower", float(np.quantile(rep_means, alpha)), "level 1: lower percentile of replicate mean risk"),
            ("mean_risk_replicates_upper", float(np.quantile(rep_means, 1 - alpha)), "level 1: upper percentile of replicate mean risk"),
            ("instability_index", float(ind["mape"].mean()), "level 4: mean absolute prediction error (MAPE) across individuals"),
            ("mape_median", float(ind["mape"].median()), "median individual MAPE"),
            ("mape_p90", float(ind["mape"].quantile(0.9)), "90th percentile of individual MAPE"),
            ("mape_max", float(ind["mape"].max()), "largest individual MAPE"),
            ("interval_width_mean", float(ind["interval_width"].mean()), "mean width of individual percentile interval"),
            ("interval_width_median", float(ind["interval_width"].median()), "median width of individual percentile interval"),
            (
                "share_wide_interval",
                float((ind["interval_width"] > self.wide_interval).mean()),
                f"share of individuals with interval wider than {self.wide_interval:g}",
            ),
        ]
        for t in self.thresholds:
            cii = ind[f"cii@{t:g}"]
            rows += [
                (f"flagged_original@{t:g}", float((p >= t).mean()), f"share flagged by the original model at {t:g}"),
                (f"cii_mean@{t:g}", float(cii.mean()), "mean classification instability index (expected share of decisions that change on refitting)"),
                (
                    f"share_flip_prone@{t:g}",
                    float((cii > self.flip_tolerance).mean()),
                    f"share of individuals whose decision changes in more than {self.flip_tolerance:.0%} of replicates",
                ),
                (f"expected_changed_decisions@{t:g}", float(cii.sum()), "expected number of individuals whose decision changes if the model is rebuilt"),
            ]
        return [{"population": population, "measure": m, "value": v, "description": d} for m, v, d in rows]

    def summary(self) -> pd.DataFrame:
        """Key stability measures for the development and (if present) external population."""
        rows = self._population_summary("development")
        if self.has_external:
            rows += self._population_summary("external")
        return pd.DataFrame(rows)

    def distribution_stability(self, population: str = "development", quantiles: Sequence[float] = (0.1, 0.25, 0.5, 0.75, 0.9)) -> pd.DataFrame:
        """Level 2: stability of the distribution of predicted risks."""
        _, p, P, _, _ = self._arrays(population)
        alpha = (1.0 - self.interval_level) / 2.0
        rep_q = np.quantile(P, quantiles, axis=1)  # (len(q), B)
        return pd.DataFrame(
            {
                "quantile": list(quantiles),
                "original": np.quantile(p, quantiles),
                "replicates_mean": rep_q.mean(axis=1),
                "replicates_lower": np.quantile(rep_q, alpha, axis=1),
                "replicates_upper": np.quantile(rep_q, 1 - alpha, axis=1),
            }
        )

    def subgroup_summary(self, population: str = "development") -> pd.DataFrame:
        """Level 3: stability within subgroups (requires ``subgroups``)."""
        ind = self.individual(population)
        if "subgroup" not in ind:
            raise ValueError("No subgroup labels were supplied for this population.")
        agg = {"n": ("risk", "size"), "mean_risk": ("risk", "mean"), "instability_index": ("mape", "mean"),
               "mape_p90": ("mape", lambda s: s.quantile(0.9)), "interval_width_mean": ("interval_width", "mean")}
        if "outcome" in ind:
            agg["events"] = ("outcome", "sum")
        for t in self.thresholds:
            agg[f"cii_mean@{t:g}"] = (f"cii@{t:g}", "mean")
            agg[f"share_flip_prone@{t:g}"] = (f"cii@{t:g}", lambda s: float((s > self.flip_tolerance).mean()))
        return ind.groupby("subgroup", dropna=False).agg(**agg).reset_index()

    def optimism(self) -> pd.DataFrame:
        """Internal validation from the bootstrap replicates.

        Columns:

        * ``apparent`` - original model evaluated on its own development data;
        * ``optimism`` / ``optimism_corrected`` - Harrell's bootstrap
          optimism correction (mean of replicate apparent minus replicate
          performance on the original data);
        * ``oob_mean`` - mean out-of-bag performance of replicate models;
        * ``estimate_632`` / ``estimate_632plus`` - Efron's .632 and .632+
          estimators. Flexible learners (forests, boosting) are often
          over-optimistic under Harrell's correction; compare with .632+.
        """
        rm = self.replicate_metrics
        if rm.empty or "bootstrap_apparent" not in set(rm["sample"]):
            raise ValueError("Optimism was not computed; run the audit with optimism=True.")
        app = rm[rm["sample"] == "bootstrap_apparent"].set_index("replicate")
        test = rm[rm["sample"] == "bootstrap_test"].set_index("replicate")
        oob = rm[rm["sample"] == "bootstrap_oob"].set_index("replicate")
        common = app.index.intersection(test.index)
        gamma = _no_information_values(self.y, self.p_original)
        rows = []
        for metric, apparent in self.performance_apparent.items():
            if metric not in app.columns:
                continue
            diff = (app.loc[common, metric] - test.loc[common, metric]).astype(float)
            opt = float(np.nanmean(diff))
            oob_mean = float(np.nanmean(oob[metric].astype(float))) if (not oob.empty and metric in oob) else np.nan
            est632 = 0.368 * apparent + 0.632 * oob_mean
            est632p = np.nan
            if metric in gamma and np.isfinite(oob_mean):
                g = gamma[metric]
                denom = g - apparent
                ror = (oob_mean - apparent) / denom if denom != 0 else 0.0
                ror = float(np.clip(ror, 0.0, 1.0)) if (oob_mean - apparent) * denom > 0 else 0.0
                w = 0.632 / (1.0 - 0.368 * ror)
                est632p = (1.0 - w) * apparent + w * oob_mean
            rows.append(
                {
                    "metric": metric,
                    "apparent": apparent,
                    "bootstrap_apparent_mean": float(np.nanmean(app.loc[common, metric])),
                    "bootstrap_test_mean": float(np.nanmean(test.loc[common, metric])),
                    "optimism": opt,
                    "optimism_corrected": apparent - opt,
                    "oob_mean": oob_mean,
                    "estimate_632": est632,
                    "estimate_632plus": est632p,
                }
            )
        return pd.DataFrame(rows)

    def performance_variability(self, population: str = "development") -> pd.DataFrame:
        """Distribution of replicate-model performance in a population.

        For ``"external"`` this shows how much a model's external performance
        depends on the particular development sample it happened to be fitted on.
        """
        sample = "bootstrap_test" if population == "development" else "external"
        rm = self.replicate_metrics
        if rm.empty or sample not in set(rm["sample"]):
            raise ValueError(f"No replicate metrics for population {population!r} (external outcomes needed).")
        sub = rm[rm["sample"] == sample].drop(columns=["replicate", "seed", "sample"])
        reference = self.performance_apparent if population == "development" else (self.performance_external_original or {})
        alpha = (1.0 - self.interval_level) / 2.0
        rows = []
        for metric in sub.columns:
            values = sub[metric].astype(float).to_numpy()
            rows.append(
                {
                    "metric": metric,
                    "original_model": reference.get(metric, np.nan),
                    "replicates_median": float(np.nanmedian(values)),
                    "replicates_lower": float(np.nanquantile(values, alpha)),
                    "replicates_upper": float(np.nanquantile(values, 1 - alpha)),
                    "replicates_sd": float(np.nanstd(values, ddof=1)),
                }
            )
        return pd.DataFrame(rows)

    def calibration_instability(
        self, population: str = "development", grid: Optional[np.ndarray] = None, method: str = "spline"
    ) -> Dict[str, np.ndarray]:
        """Smoothed calibration curves of every replicate model in a population
        with observed outcomes, plus the original model's curve."""
        y, p, P, _, _ = self._arrays(population)
        if y is None:
            raise ValueError("Calibration needs observed outcomes for this population.")
        if grid is None:
            grid = np.linspace(0.01, 0.99, 99)
        original = calibration_curve(y, p, grid=grid, method=method)["observed"]
        curves = np.vstack([calibration_curve(y, P[b].astype(float), grid=grid, method=method)["observed"] for b in range(P.shape[0])])
        return {"predicted": np.asarray(grid), "original": original, "replicates": curves}

    def decision_curve_instability(
        self, population: str = "development", thresholds: Optional[Sequence[float]] = None
    ) -> pd.DataFrame:
        """Net benefit of the original and every replicate model across thresholds."""
        y, p, P, _, _ = self._arrays(population)
        if y is None:
            raise ValueError("Net benefit needs observed outcomes for this population.")
        if thresholds is None:
            upper = min(0.99, max(0.5, float(np.quantile(p, 0.99))))
            thresholds = np.round(np.linspace(0.01, upper, 50), 4)
        thresholds = np.asarray(thresholds, dtype=float)
        frames = [
            pd.DataFrame({"replicate": -1, "strategy": "original", "threshold": thresholds, "net_benefit": net_benefit(y, p, thresholds)}),
            pd.DataFrame({"replicate": -1, "strategy": "treat_all", "threshold": thresholds, "net_benefit": net_benefit_treat_all(y, thresholds)}),
            pd.DataFrame({"replicate": -1, "strategy": "treat_none", "threshold": thresholds, "net_benefit": 0.0}),
        ]
        for b in range(P.shape[0]):
            frames.append(pd.DataFrame({"replicate": b, "strategy": "replicate", "threshold": thresholds, "net_benefit": net_benefit(y, P[b], thresholds)}))
        return pd.concat(frames, ignore_index=True)

    # ------------------------------------------------------ new individuals
    def predict_interval(self, X_new: Any, level: Optional[float] = None) -> pd.DataFrame:
        """Risk for new individuals with an instability interval from the
        stored replicate models (requires ``keep_models=True``)."""
        if not self.models or self.builder is None or self.original_model is None:
            raise ValueError("Replicate models were not stored; rerun the audit with keep_models=True.")
        level = self.interval_level if level is None else float(level)
        alpha = (1.0 - level) / 2.0
        X_new = check_features(X_new, "X_new")
        p = self.builder.predict_risk(self.original_model, X_new)
        P = np.vstack([self.builder.predict_risk(m, X_new) for m in self.models])
        lower, upper = np.quantile(P, [alpha, 1 - alpha], axis=0)
        out = pd.DataFrame({"risk": p, "risk_lower": lower, "risk_upper": upper})
        out["interval_width"] = out["risk_upper"] - out["risk_lower"]
        out["mape"] = np.mean(np.abs(P - p[None, :]), axis=0)
        for t in self.thresholds:
            out[f"cii@{t:g}"] = np.mean((P >= t) != (p >= t)[None, :], axis=0)
        return out

    # --------------------------------------------------------------- export
    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable summary (no individual-level data)."""
        out: Dict[str, Any] = {
            "config": self.config,
            "summary": self.summary().to_dict(orient="records"),
            "performance_apparent": self.performance_apparent,
            "failures": self.failures,
        }
        try:
            out["optimism"] = self.optimism().to_dict(orient="records")
        except ValueError:
            pass
        if self.performance_external_original is not None:
            out["performance_external_original"] = self.performance_external_original
            out["external_performance_variability"] = self.performance_variability("external").to_dict(orient="records")
        if self.subgroups is not None:
            out["subgroups_development"] = self.subgroup_summary("development").to_dict(orient="records")
        return _jsonable(out)

    def to_json(self, path: Optional[str] = None, indent: int = 2) -> str:
        text = json.dumps(self.to_dict(), indent=indent)
        if path is not None:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(text)
        return text

    def export(self, directory: str, include_individual: bool = True) -> Dict[str, str]:
        """Write CSV/JSON outputs (and an HTML report if matplotlib is
        available) to ``directory``. Returns the paths written."""
        import os

        os.makedirs(directory, exist_ok=True)
        paths: Dict[str, str] = {}

        def write(name: str, frame: pd.DataFrame) -> None:
            path = os.path.join(directory, name)
            frame.to_csv(path, index=False)
            paths[name] = path

        write("summary.csv", self.summary())
        write("distribution_stability.csv", self.distribution_stability())
        write("replicate_metrics.csv", self.replicate_metrics)
        try:
            write("optimism.csv", self.optimism())
        except ValueError:
            pass
        if include_individual:
            write("individual_development.csv", self.individual("development"))
        if self.subgroups is not None:
            write("subgroups_development.csv", self.subgroup_summary("development"))
        if self.has_external:
            write("distribution_stability_external.csv", self.distribution_stability("external"))
            if include_individual:
                write("individual_external.csv", self.individual("external"))
            if self.subgroups_external is not None:
                write("subgroups_external.csv", self.subgroup_summary("external"))
            if self.y_external is not None:
                write("performance_external.csv", self.performance_variability("external"))
        json_path = os.path.join(directory, "result.json")
        self.to_json(json_path)
        paths["result.json"] = json_path
        try:
            from .report import write_html_report

            html = os.path.join(directory, "report.html")
            write_html_report(self, html)
            paths["report.html"] = html
        except ImportError:
            pass
        return paths

    def save(self, path: str) -> None:
        """Persist the full result (including stored models) with joblib."""
        import joblib

        joblib.dump(self, path, compress=3)

    @staticmethod
    def load(path: str) -> "StabilityResult":
        import joblib

        obj = joblib.load(path)
        if not isinstance(obj, StabilityResult):
            raise TypeError("The file does not contain a StabilityResult.")
        return obj

    def report(self, path: str, title: str = "Prediction instability audit") -> str:
        from .report import write_html_report

        return write_html_report(self, path, title=title)

    # ------------------------------------------------------------- plotting
    def plot_instability(self, population: str = "development", **kwargs: Any):
        from .plotting import plot_prediction_instability

        return plot_prediction_instability(self, population=population, **kwargs)

    def plot_mape(self, population: str = "development", **kwargs: Any):
        from .plotting import plot_mape

        return plot_mape(self, population=population, **kwargs)

    def plot_classification_instability(self, threshold: Optional[float] = None, population: str = "development", **kwargs: Any):
        from .plotting import plot_classification_instability

        return plot_classification_instability(self, threshold=threshold, population=population, **kwargs)

    def plot_calibration_instability(self, population: str = "development", **kwargs: Any):
        from .plotting import plot_calibration_instability

        return plot_calibration_instability(self, population=population, **kwargs)

    def plot_decision_curve_instability(self, population: str = "development", **kwargs: Any):
        from .plotting import plot_decision_curve_instability

        return plot_decision_curve_instability(self, population=population, **kwargs)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        ext = f", external n={self.P_external.shape[1]}" if self.has_external else ""
        return (
            f"StabilityResult(model={self.config.get('model')!r}, replicates={self.n_replicates}, "
            f"development n={self.P.shape[1]}{ext}, instability_index={self.instability_index():.4f})"
        )


def _no_information_values(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    """Performance expected when predictions carry no information about the
    outcome (every prediction paired with every outcome); used by .632+."""
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12)
    ybar = y.mean()
    return {
        "auroc": 0.5,
        "brier": float(ybar - 2 * ybar * p.mean() + np.mean(p**2)),
        "log_loss": float(-(ybar * np.mean(np.log(p)) + (1 - ybar) * np.mean(np.log1p(-p)))),
    }


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        return None if not np.isfinite(value) else value
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj
