"""Turn an audit into an explicit, reviewable acceptance decision.

A :class:`StabilityPolicy` collects limits that a model-governance group (an
AI oversight committee, a model-risk function or a quality team) agrees in
advance. :meth:`StabilityPolicy.evaluate` checks a
:class:`~cpmaudit.stability.StabilityResult` against them and returns a
pass/fail table that can be stored with the model's documentation or used as
a gate in an automated deployment pipeline.

The default limits are placeholders that make the mechanism usable out of the
box; they are not recommendations. Set them for the clinical decision at hand.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from .stability import StabilityResult


@dataclass
class StabilityPolicy:
    """Acceptance limits for prediction instability.

    Any limit set to ``None`` is not checked.

    Attributes
    ----------
    max_instability_index:
        Upper limit for the mean absolute prediction error (MAPE) across individuals.
    max_mape_p90:
        Upper limit for the 90th percentile of individual MAPE.
    max_share_wide_interval:
        Upper limit for the share of individuals whose instability interval is
        wider than ``wide_interval``.
    wide_interval:
        Interval width regarded as too uncertain for an individual decision.
    max_share_flip_prone:
        Upper limit, at every decision threshold, for the share of individuals
        whose decision changes in more than ``flip_tolerance`` of replicates.
    flip_tolerance:
        Classification instability index above which an individual is flip-prone.
    max_abs_calibration_slope_optimism:
        Upper limit for the absolute optimism in the calibration slope.
    min_replicates:
        Minimum number of successful bootstrap replicates.
    populations:
        Populations to check (``"development"`` and/or ``"external"``).
    """

    max_instability_index: Optional[float] = 0.02
    max_mape_p90: Optional[float] = 0.05
    max_share_wide_interval: Optional[float] = 0.10
    wide_interval: float = 0.10
    max_share_flip_prone: Optional[float] = 0.05
    flip_tolerance: float = 0.10
    max_abs_calibration_slope_optimism: Optional[float] = 0.10
    min_replicates: Optional[int] = 200
    populations: List[str] = field(default_factory=lambda: ["development"])

    def evaluate(self, result: StabilityResult) -> "PolicyReport":
        result.flip_tolerance = self.flip_tolerance
        result.wide_interval = self.wide_interval
        summ = result.summary()
        checks: List[Dict[str, Any]] = []

        def check(name: str, value: float, limit: Optional[float], population: str, larger_is_worse: bool = True):
            if limit is None:
                return
            passed = value <= limit if larger_is_worse else value >= limit
            checks.append({"population": population, "check": name, "value": value, "limit": limit, "passed": bool(passed)})

        if self.min_replicates is not None:
            check("replicates", float(result.n_replicates), float(self.min_replicates), "all", larger_is_worse=False)
        for pop in self.populations:
            if pop == "external" and not result.has_external:
                continue
            values = summ[summ["population"] == pop].set_index("measure")["value"]
            check("instability_index", float(values["instability_index"]), self.max_instability_index, pop)
            check("mape_p90", float(values["mape_p90"]), self.max_mape_p90, pop)
            check("share_wide_interval", float(values["share_wide_interval"]), self.max_share_wide_interval, pop)
            for t in result.thresholds:
                check(f"share_flip_prone@{t:g}", float(values[f"share_flip_prone@{t:g}"]), self.max_share_flip_prone, pop)
        if self.max_abs_calibration_slope_optimism is not None:
            try:
                opt = result.optimism().set_index("metric")
                if "calibration_slope" in opt.index:
                    check(
                        "abs_calibration_slope_optimism",
                        abs(float(opt.loc["calibration_slope", "optimism"])),
                        self.max_abs_calibration_slope_optimism,
                        "development",
                    )
            except ValueError:
                pass
        table = pd.DataFrame(checks)
        return PolicyReport(passed=bool(table["passed"].all()) if not table.empty else True, checks=table, policy=asdict(self))


@dataclass
class PolicyReport:
    passed: bool
    checks: pd.DataFrame
    policy: Dict[str, Any]

    def failed(self) -> pd.DataFrame:
        return self.checks[~self.checks["passed"]]

    def raise_if_failed(self) -> None:
        """Raise an ``AssertionError`` listing failed checks (for CI pipelines)."""
        if not self.passed:
            rows = self.failed()
            details = "; ".join(f"{r.population}/{r.check}={r.value:.4g} (limit {r.limit:.4g})" for r in rows.itertuples())
            raise AssertionError(f"Stability policy failed: {details}")

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        status = "PASSED" if self.passed else "FAILED"
        return f"Stability policy {status}\n" + self.checks.to_string(index=False)


def model_facts(result: StabilityResult) -> Dict[str, Any]:
    """Compact facts for a model card or model inventory entry."""
    summ = result.summary()
    dev = summ[summ["population"] == "development"].set_index("measure")["value"]
    facts: Dict[str, Any] = {
        "model": result.config.get("model"),
        "development_n": result.config.get("n_development"),
        "development_events": result.config.get("events_development"),
        "bootstrap": result.config.get("bootstrap"),
        "replicates": result.n_replicates,
        "instability_index": float(dev["instability_index"]),
        "mape_p90": float(dev["mape_p90"]),
        "interval_width_median": float(dev["interval_width_median"]),
        "thresholds": {},
        "package_version": result.config.get("package_version"),
    }
    for t in result.thresholds:
        facts["thresholds"][f"{t:g}"] = {
            "flagged_share": float(dev[f"flagged_original@{t:g}"]),
            "mean_cii": float(dev[f"cii_mean@{t:g}"]),
            "share_flip_prone": float(dev[f"share_flip_prone@{t:g}"]),
        }
    try:
        opt = result.optimism().set_index("metric")
        facts["optimism_corrected"] = {
            m: float(opt.loc[m, "optimism_corrected"])
            for m in ("auroc", "calibration_slope", "calibration_intercept", "brier")
            if m in opt.index
        }
    except ValueError:
        pass
    if result.has_external:
        ext = summ[summ["population"] == "external"].set_index("measure")["value"]
        facts["external_instability_index"] = float(ext["instability_index"])
    return facts
