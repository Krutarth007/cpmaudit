"""Matplotlib figures for stability audits (requires ``matplotlib``)."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

REPLICATE = "#2a78d6"
ORIGINAL = "#eb6834"
REFERENCE = "#8a8984"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"


def _axes(ax: Any, figsize=(6.0, 4.5)):
    if ax is not None:
        return ax
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("Plotting requires matplotlib: `pip install cpmaudit[plot]`.") from exc
    _, ax = plt.subplots(figsize=figsize)
    return ax


def _style(ax: Any, xlabel: str, ylabel: str, title: Optional[str]) -> None:
    ax.set_xlabel(xlabel, color=INK_SECONDARY)
    ax.set_ylabel(ylabel, color=INK_SECONDARY)
    if title:
        ax.set_title(title, color=INK, loc="left", fontsize=11)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(REFERENCE)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)


def _subset(n: int, max_points: int, seed: int = 0) -> np.ndarray:
    if n <= max_points:
        return np.arange(n)
    return np.sort(np.random.default_rng(seed).choice(n, size=max_points, replace=False))


def _rolling(x: np.ndarray, y: np.ndarray, window: int):
    order = np.argsort(x)
    xs, ys = x[order], y[order]
    if xs.size <= window:
        return xs, ys
    kernel = np.ones(window) / window
    return np.convolve(xs, kernel, mode="valid"), np.convolve(ys, kernel, mode="valid")


def plot_prediction_instability(result, population: str = "development", max_individuals: int = 1000, ax: Any = None, title: Optional[str] = None):
    """Risk from each replicate model (y) against the original risk (x)."""
    ax = _axes(ax)
    _, p, P, _, _ = result._arrays(population)
    idx = _subset(p.size, max_individuals)
    xs = np.repeat(p[idx][None, :], P.shape[0], axis=0).ravel()
    ys = P[:, idx].ravel()
    ax.scatter(xs, ys, s=2, alpha=min(0.3, 30.0 / P.shape[0]), color=REPLICATE, linewidths=0, rasterized=True, label="replicate models")
    ind = result.individual(population)
    window = max(5, p.size // 50)
    for col in ("risk_lower", "risk_upper"):
        bx, by = _rolling(ind["risk"].to_numpy(), ind[col].to_numpy(), window)
        ax.plot(bx, by, color=ORIGINAL, linewidth=2, label=f"{result.interval_level:.0%} interval" if col == "risk_lower" else None)
    lim = [0, max(float(np.max(ys)), float(np.max(p))) * 1.02]
    ax.plot(lim, lim, color=REFERENCE, linewidth=1, linestyle="--")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    _style(ax, "Risk from original model", "Risk from bootstrap models", title or f"Prediction instability ({population})")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    return ax


def plot_mape(result, population: str = "development", max_individuals: int = 5000, ax: Any = None, title: Optional[str] = None):
    """Individual mean absolute prediction error against original risk."""
    ax = _axes(ax)
    ind = result.individual(population)
    idx = _subset(len(ind), max_individuals)
    ax.scatter(ind["risk"].to_numpy()[idx], ind["mape"].to_numpy()[idx], s=8, alpha=0.5, color=REPLICATE, linewidths=0, rasterized=True)
    _style(ax, "Risk from original model", "Mean absolute prediction error", title or f"Individual MAPE ({population})")
    return ax


def plot_classification_instability(result, threshold: Optional[float] = None, population: str = "development", max_individuals: int = 5000, ax: Any = None, title: Optional[str] = None):
    """Classification instability index (share of replicate models with a
    different decision) against original risk."""
    if not result.thresholds:
        raise ValueError("The audit was run without decision thresholds.")
    t = result.thresholds[0] if threshold is None else float(threshold)
    ax = _axes(ax)
    ind = result.individual(population)
    col = f"cii@{t:g}"
    if col not in ind:
        raise ValueError(f"Threshold {t} was not part of the audit.")
    idx = _subset(len(ind), max_individuals)
    ax.scatter(ind["risk"].to_numpy()[idx], ind[col].to_numpy()[idx], s=8, alpha=0.5, color=REPLICATE, linewidths=0, rasterized=True)
    ax.axvline(t, color=ORIGINAL, linewidth=2, label=f"threshold {t:g}")
    ax.axhline(result.flip_tolerance, color=REFERENCE, linewidth=1, linestyle="--", label=f"flip tolerance {result.flip_tolerance:g}")
    ax.set_ylim(-0.02, 1.02)
    _style(ax, "Risk from original model", "Classification instability index", title or f"Decision instability at {t:g} ({population})")
    ax.legend(frameon=False, fontsize=8)
    return ax


def plot_calibration_instability(result, population: str = "development", method: str = "spline", ax: Any = None, title: Optional[str] = None):
    """Smoothed calibration curve of every replicate model and the original model."""
    ax = _axes(ax)
    cal = result.calibration_instability(population, method=method)
    grid = cal["predicted"]
    for i, curve in enumerate(cal["replicates"]):
        ax.plot(grid, curve, color=REPLICATE, alpha=min(0.25, 25.0 / len(cal["replicates"])), linewidth=1, label="replicate models" if i == 0 else None)
    ax.plot(grid, cal["original"], color=ORIGINAL, linewidth=2, label="original model")
    finite = np.isfinite(cal["original"])
    hi = float(np.nanmax(grid[finite])) if finite.any() else 1.0
    ax.plot([0, hi], [0, hi], color=REFERENCE, linewidth=1, linestyle="--", label="ideal")
    ax.set_xlim(0, hi)
    ax.set_ylim(0, min(1.0, hi * 1.3))
    _style(ax, "Predicted risk", "Observed risk (smoothed)", title or f"Calibration instability ({population})")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    return ax


def plot_decision_curve_instability(result, population: str = "development", thresholds=None, ax: Any = None, title: Optional[str] = None):
    """Net benefit of every replicate model, the original model, and the
    treat-all / treat-none strategies."""
    ax = _axes(ax)
    dc = result.decision_curve_instability(population, thresholds)
    reps = dc[dc["strategy"] == "replicate"]
    n_rep = reps["replicate"].nunique()
    for i, (_, grp) in enumerate(reps.groupby("replicate")):
        ax.plot(grp["threshold"], grp["net_benefit"], color=REPLICATE, alpha=min(0.25, 25.0 / max(n_rep, 1)), linewidth=1, label="replicate models" if i == 0 else None)
    orig = dc[dc["strategy"] == "original"]
    ax.plot(orig["threshold"], orig["net_benefit"], color=ORIGINAL, linewidth=2, label="original model")
    allp = dc[dc["strategy"] == "treat_all"]
    ax.plot(allp["threshold"], allp["net_benefit"], color=INK_SECONDARY, linewidth=1, linestyle="--", label="treat all")
    ax.axhline(0, color=REFERENCE, linewidth=1, label="treat none")
    top = float(np.nanmax(orig["net_benefit"])) if len(orig) else 0.1
    ax.set_ylim(-0.05 * max(top, 0.01), max(top, 0.01) * 1.25)
    _style(ax, "Threshold probability", "Net benefit", title or f"Decision curve instability ({population})")
    ax.legend(frameon=False, fontsize=8)
    return ax


def plot_instability_curve(curve, measure: str = "instability_index", ax: Any = None, title: Optional[str] = None):
    """Instability measure against development sample size (log scale)."""
    ax = _axes(ax)
    agg = curve.table.groupby("size")[measure].agg(["mean", "min", "max"]).reset_index()
    ax.plot(agg["size"], agg["mean"], color=REPLICATE, linewidth=2, marker="o", markersize=6)
    ax.fill_between(agg["size"], agg["min"], agg["max"], color=REPLICATE, alpha=0.15, linewidth=0)
    ax.set_xscale("log")
    _style(ax, "Development sample size", measure.replace("_", " "), title or "Instability by sample size")
    return ax
