"""Self-contained HTML report for a stability audit (figures embedded)."""

from __future__ import annotations

import base64
import html
import io
from typing import Any, Callable, List, Optional

import numpy as np
import pandas as pd

from .stability import StabilityResult

_CSS = """
body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;color:#0b0b0b;background:#fcfcfb;
max-width:1100px;margin:32px auto;padding:0 20px;line-height:1.45}
h1{font-size:24px;margin-bottom:4px}h2{font-size:18px;margin-top:36px;border-bottom:1px solid #e4e3df;padding-bottom:4px}
p.meta{color:#52514e;margin-top:0}table{border-collapse:collapse;margin:8px 0 16px;font-size:13px}
th,td{border-bottom:1px solid #e4e3df;padding:4px 10px;text-align:right}th{background:#f3f2ef;color:#52514e}
td:first-child,th:first-child{text-align:left}.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}
figure{margin:0}figure img{width:100%;border:1px solid #e4e3df;border-radius:4px}
figcaption{font-size:12px;color:#52514e}.note{font-size:13px;color:#52514e}
dl{font-size:13px}dt{font-weight:600;margin-top:8px}dd{margin-left:0;color:#52514e}
"""

_GLOSSARY = [
    ("Original model", "The model fitted on all development data, as it would be deployed."),
    ("Bootstrap (replicate) models", "Models built with the same full procedure on resamples of the development data. "
     "Their disagreement with the original model shows how much predictions depend on the particular sample."),
    ("Instability index (MAPE)", "Average absolute difference between an individual's risk from the replicate models "
     "and from the original model, averaged over individuals."),
    ("Individual interval", "Percentile range of an individual's risk across replicate models."),
    ("Classification instability index (CII)", "For one individual and one threshold: the share of replicate models "
     "whose decision (above or below the threshold) differs from the original model's decision."),
    ("Flip-prone individual", "An individual whose CII exceeds the flip tolerance; acting on that individual's "
     "prediction alone is not well supported by the data."),
    ("Optimism", "Average amount by which performance measured on the data used for fitting overstates performance on "
     "new data drawn from the same population."),
]


def _fmt(value: Any) -> str:
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return "&ndash;"
        if value != 0 and (abs(value) >= 1e5 or abs(value) < 1e-3):
            return f"{value:.3e}"
        if float(value).is_integer() and abs(value) >= 1:
            return f"{int(value):,}"
        return f"{value:.4f}"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    return html.escape(str(value))


def _table(frame: pd.DataFrame, max_rows: int = 200) -> str:
    frame = frame.head(max_rows)
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in frame.columns)
    body = "".join("<tr>" + "".join(f"<td>{_fmt(v)}</td>" for v in row) + "</tr>" for row in frame.itertuples(index=False))
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _figure(draw: Callable[[Any], Any], caption: str) -> str:
    try:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
    except ImportError:
        return ""
    fig = Figure(figsize=(6.0, 4.4), dpi=110)
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    try:
        draw(ax)
    except Exception as exc:  # noqa: BLE001 - a failed figure must not break the report
        return f"<p class='note'>Figure unavailable: {html.escape(str(exc))}</p>"
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    data = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"<figure><img alt='{html.escape(caption)}' src='data:image/png;base64,{data}'/><figcaption>{html.escape(caption)}</figcaption></figure>"


def write_html_report(result: StabilityResult, path: str, title: str = "Prediction instability audit") -> str:
    """Write a standalone HTML report and return its path."""
    from . import plotting

    cfg = result.config
    parts: List[str] = [
        "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>",
        f"<title>{html.escape(title)}</title><style>{_CSS}</style></head><body>",
        f"<h1>{html.escape(title)}</h1>",
        f"<p class='meta'>Model: {html.escape(str(cfg.get('model')))} &middot; bootstrap: {html.escape(str(cfg.get('bootstrap')))} "
        f"&middot; {result.n_replicates} replicates &middot; generated {html.escape(str(cfg.get('created')))} "
        f"with cpmaudit {html.escape(str(cfg.get('package_version')))}</p>",
    ]

    summ = result.summary()
    for population in summ["population"].unique():
        parts.append(f"<h2>Stability summary: {population}</h2>")
        parts.append(_table(summ[summ["population"] == population][["measure", "value", "description"]]))

    parts.append("<h2>Figures: development population</h2><div class='grid'>")
    parts.append(_figure(lambda ax: plotting.plot_prediction_instability(result, "development", ax=ax), "Risk from bootstrap models against risk from the original model"))
    parts.append(_figure(lambda ax: plotting.plot_mape(result, "development", ax=ax), "Individual mean absolute prediction error"))
    if result.thresholds:
        for t in result.thresholds:
            parts.append(_figure(lambda ax, t=t: plotting.plot_classification_instability(result, t, "development", ax=ax), f"Decision instability at threshold {t:g}"))
    parts.append(_figure(lambda ax: plotting.plot_calibration_instability(result, "development", ax=ax), "Calibration curves of replicate models in the development data"))
    parts.append(_figure(lambda ax: plotting.plot_decision_curve_instability(result, "development", ax=ax), "Net benefit of replicate models in the development data"))
    parts.append("</div>")

    if result.has_external:
        parts.append("<h2>Figures: external population</h2><div class='grid'>")
        parts.append(_figure(lambda ax: plotting.plot_prediction_instability(result, "external", ax=ax), "Risk from bootstrap models against the original model (external)"))
        if result.thresholds:
            t0 = result.thresholds[0]
            parts.append(_figure(lambda ax: plotting.plot_classification_instability(result, t0, "external", ax=ax), f"Decision instability at {t0:g} (external)"))
        if result.y_external is not None:
            parts.append(_figure(lambda ax: plotting.plot_calibration_instability(result, "external", ax=ax), "Calibration curves of replicate models (external)"))
            parts.append(_figure(lambda ax: plotting.plot_decision_curve_instability(result, "external", ax=ax), "Net benefit of replicate models (external)"))
        parts.append("</div>")

    parts.append("<h2>Distribution of predicted risks</h2>")
    parts.append(_table(result.distribution_stability("development")))
    try:
        parts.append("<h2>Optimism-corrected performance (development)</h2>")
        parts.append(_table(result.optimism()))
    except ValueError:
        parts.append("<p class='note'>Optimism correction was not requested.</p>")
    if result.y_external is not None:
        parts.append("<h2>External performance: original model and spread across replicate models</h2>")
        parts.append(_table(result.performance_variability("external")))
    if result.subgroups is not None:
        parts.append("<h2>Subgroup stability (development)</h2>")
        parts.append(_table(result.subgroup_summary("development")))
    if result.subgroups_external is not None:
        parts.append("<h2>Subgroup stability (external)</h2>")
        parts.append(_table(result.subgroup_summary("external")))

    parts.append("<h2>Configuration</h2>")
    parts.append(_table(pd.DataFrame({"setting": list(cfg), "value": [str(v) for v in cfg.values()]})))
    if result.failures:
        parts.append(f"<p class='note'>{len(result.failures)} replicate(s) failed and were excluded.</p>")
    parts.append("<h2>How to read this report</h2><dl>")
    for term, text in _GLOSSARY:
        parts.append(f"<dt>{html.escape(term)}</dt><dd>{html.escape(text)}</dd>")
    parts.append("</dl></body></html>")

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))
    return path


def report_to_string(result: StabilityResult, title: Optional[str] = None) -> str:
    """Return the HTML report as a string (useful for notebooks)."""
    import os
    import tempfile

    fd, tmp = tempfile.mkstemp(suffix=".html")
    os.close(fd)
    try:
        write_html_report(result, tmp, title=title or "Prediction instability audit")
        with open(tmp, encoding="utf-8") as fh:
            return fh.read()
    finally:
        os.remove(tmp)
