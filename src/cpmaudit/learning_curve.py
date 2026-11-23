"""How does individual-level instability shrink as the development sample grows?

:func:`instability_curve` repeatedly subsamples the development data at a
series of sizes, runs a :class:`~cpmaudit.stability.StabilityAudit` on each
subsample, and measures instability on a *fixed reference panel* of
individuals so that results are comparable across sizes. The output supports
an empirical, model-specific answer to "how many patients do we need before
individual predictions are stable enough to act on?".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from ._validation import (
    as_binary_outcome,
    as_label_array,
    as_thresholds,
    check_features,
    check_same_length,
    take_rows,
)
from .resampling import subsample_indices
from .stability import StabilityAudit


@dataclass
class InstabilityCurve:
    table: pd.DataFrame
    thresholds: tuple
    config: Dict[str, Any] = field(default_factory=dict)

    def aggregate(self) -> pd.DataFrame:
        """Mean and range over repeats for every measure at each size."""
        numeric = self.table.drop(columns=["repeat"]).groupby("size")
        return numeric.agg(["mean", "min", "max"]).reset_index()

    def minimum_size(self, measure: str = "instability_index", target: float = 0.02, lower_is_better: bool = True) -> Optional[int]:
        """Smallest evaluated size whose mean ``measure`` meets ``target``
        (and stays within target for all larger sizes)."""
        means = self.table.groupby("size")[measure].mean().sort_index()
        ok = means <= target if lower_is_better else means >= target
        sizes = means.index.to_numpy()
        for i, size in enumerate(sizes):
            if ok.iloc[i:].all():
                return int(size)
        return None

    def plot(self, measure: str = "instability_index", ax: Any = None):
        from .plotting import plot_instability_curve

        return plot_instability_curve(self, measure=measure, ax=ax)


def instability_curve(
    model: Any,
    X: Any,
    y: Any,
    sizes: Sequence[int],
    n_bootstrap: int = 100,
    n_repeats: int = 3,
    thresholds: Iterable[float] = (0.1,),
    bootstrap: str = "simple",
    groups: Any = None,
    X_reference: Any = None,
    y_reference: Any = None,
    reference_size: int = 5000,
    random_state: Optional[int] = None,
    n_jobs: int = 1,
    verbose: int = 0,
) -> InstabilityCurve:
    """Instability of individual predictions as a function of development size.

    Parameters
    ----------
    model:
        Model specification (see :mod:`cpmaudit.models`).
    X, y:
        Pool of development data from which subsamples are drawn.
    sizes:
        Development sample sizes to evaluate.
    n_bootstrap, n_repeats:
        Bootstrap replicates per audit and independent subsamples per size.
    groups:
        Cluster labels; subsamples and bootstraps then keep clusters intact.
    X_reference, y_reference:
        Fixed panel on which instability is measured. If omitted, a panel of
        ``reference_size`` rows is held out from ``X`` and never used for
        development.
    """
    X = check_features(X, "X")
    y = as_binary_outcome(y, "y")
    check_same_length(X, y, names=("X", "y"))
    thresholds = as_thresholds(thresholds)
    rng = np.random.default_rng(random_state)
    groups_arr = as_label_array(groups, y.size, "groups") if groups is not None else None

    pool = np.arange(y.size)
    if X_reference is None:
        ref_idx = subsample_indices(y.size, min(reference_size, y.size // 3), rng, y=y, groups=groups_arr)
        in_ref = np.zeros(y.size, dtype=bool)
        in_ref[ref_idx] = True
        if groups_arr is not None:
            in_ref |= np.isin(groups_arr, np.unique(groups_arr[ref_idx]))
        X_ref, y_ref = take_rows(X, np.flatnonzero(in_ref)), y[in_ref]
        pool = np.flatnonzero(~in_ref)
    else:
        X_ref = check_features(X_reference, "X_reference")
        y_ref = as_binary_outcome(y_reference, "y_reference") if y_reference is not None else None

    sizes = sorted(int(s) for s in sizes)
    if sizes[-1] > pool.size:
        raise ValueError(f"Largest size {sizes[-1]} exceeds the {pool.size} rows available for development.")

    rows = []
    for size in sizes:
        for r in range(n_repeats):
            seed = int(rng.integers(0, 2**31 - 1))
            sub_rng = np.random.default_rng(seed)
            local = subsample_indices(
                pool.size, size, sub_rng, y=y[pool], groups=groups_arr[pool] if groups_arr is not None else None
            )
            idx = pool[local]
            audit = StabilityAudit(
                model,
                n_bootstrap=n_bootstrap,
                bootstrap=bootstrap,
                thresholds=thresholds,
                optimism=False,
                n_jobs=n_jobs,
                random_state=seed,
                smooth_calibration=False,
            )
            res = audit.run(
                take_rows(X, idx),
                y[idx],
                groups=groups_arr[idx] if (groups_arr is not None and bootstrap == "cluster") else None,
                X_external=X_ref,
                y_external=y_ref,
            )
            summ = res.summary()
            ext = summ[summ["population"] == "external"].set_index("measure")["value"]
            row = {
                "size": size,
                "repeat": r,
                "n_development": int(idx.size),
                "events_development": int(y[idx].sum()),
                "replicates": res.n_replicates,
                "instability_index": ext["instability_index"],
                "mape_p90": ext["mape_p90"],
                "interval_width_mean": ext["interval_width_mean"],
                "share_wide_interval": ext["share_wide_interval"],
            }
            for t in thresholds:
                row[f"cii_mean@{t:g}"] = ext[f"cii_mean@{t:g}"]
                row[f"share_flip_prone@{t:g}"] = ext[f"share_flip_prone@{t:g}"]
            if res.performance_external_original is not None:
                perf = res.performance_external_original
                row["reference_auroc"] = perf.get("auroc")
                row["reference_calibration_slope"] = perf.get("calibration_slope")
                row["reference_oe_ratio"] = perf.get("oe_ratio")
            rows.append(row)
            if verbose:
                print(f"[cpmaudit] size={size} repeat={r + 1}/{n_repeats} instability_index={row['instability_index']:.4f}")

    config = {
        "model": audit.builder.description,
        "n_bootstrap": n_bootstrap,
        "n_repeats": n_repeats,
        "bootstrap": bootstrap,
        "reference_n": int(y_ref.size) if y_ref is not None else None,
        "random_state": random_state,
    }
    return InstabilityCurve(table=pd.DataFrame(rows), thresholds=thresholds, config=config)
