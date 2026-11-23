"""Command-line interface.

Examples
--------
Audit a model on a CSV/Parquet table::

    cpmaudit stability --data cohort.parquet --outcome died --group-col subject_id \
        --bootstrap cluster --model logistic --bootstraps 200 --threshold 0.1 0.2 --out-dir audit_out

Minimum sample size::

    cpmaudit sampsize --parameters 24 --prevalence 0.12 --cstatistic 0.80

List the tables found in a data release::

    cpmaudit tables /path/to/data/root

Try everything on synthetic data::

    cpmaudit demo --out-dir demo_out
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

import pandas as pd

from ._version import __version__


def _read_any(path: str) -> pd.DataFrame:
    low = path.lower()
    if low.endswith((".parquet", ".pq")):
        return pd.read_parquet(path)
    if low.endswith((".tsv", ".tsv.gz")):
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


def _features(frame: pd.DataFrame, outcome: str, features: Optional[List[str]], exclude: List[str]) -> List[str]:
    if features:
        missing = [f for f in features if f not in frame.columns]
        if missing:
            raise SystemExit(f"Feature columns not found: {missing}")
        return features
    drop = set(exclude) | {outcome}
    return [c for c in frame.columns if c not in drop]


def _cmd_stability(args: argparse.Namespace) -> int:
    from .governance import StabilityPolicy
    from .stability import StabilityAudit

    dev = _read_any(args.data)
    exclude = list(args.exclude or [])
    for col in (args.group_col, args.subgroup_col, args.id_col):
        if col:
            exclude.append(col)
    feats = _features(dev, args.outcome, args.features, exclude)
    ext = _read_any(args.external) if args.external else None

    audit = StabilityAudit(
        model=args.model,
        n_bootstrap=args.bootstraps,
        bootstrap=args.bootstrap,
        thresholds=args.threshold,
        interval_level=args.interval_level,
        keep_models=False,
        n_jobs=args.n_jobs,
        random_state=args.seed,
        verbose=1,
    )
    result = audit.run(
        dev[feats],
        dev[args.outcome],
        groups=dev[args.group_col] if args.group_col else None,
        subgroups=dev[args.subgroup_col] if args.subgroup_col else None,
        ids=dev[args.id_col] if args.id_col else None,
        X_external=ext[feats] if ext is not None else None,
        y_external=ext[args.outcome] if (ext is not None and args.outcome in ext.columns) else None,
        subgroups_external=ext[args.subgroup_col] if (ext is not None and args.subgroup_col and args.subgroup_col in ext.columns) else None,
        ids_external=ext[args.id_col] if (ext is not None and args.id_col and args.id_col in ext.columns) else None,
    )
    result.flip_tolerance = args.flip_tolerance
    paths = result.export(args.out_dir, include_individual=not args.no_individual)
    print(result.summary().to_string(index=False))
    if args.policy:
        report = StabilityPolicy(flip_tolerance=args.flip_tolerance, min_replicates=min(200, args.bootstraps)).evaluate(result)
        print(str(report))
        report.checks.to_csv(f"{args.out_dir.rstrip('/')}/policy_checks.csv", index=False)
        if not report.passed:
            print("Outputs written:", ", ".join(paths))
            return 2
    print("Outputs written:", ", ".join(paths))
    return 0


def _cmd_sampsize(args: argparse.Namespace) -> int:
    from .sampsize import minimum_sample_size_binary

    res = minimum_sample_size_binary(
        n_parameters=args.parameters,
        prevalence=args.prevalence,
        r2_cox_snell=args.r2_cs,
        c_statistic=args.cstatistic,
        r2_nagelkerke=args.r2_nagelkerke,
        shrinkage=args.shrinkage,
    )
    print(res.criteria.to_string(index=False))
    print(
        f"\nMinimum sample size: {res.minimum_n} ({res.minimum_events:.1f} events, "
        f"{res.events_per_parameter:.2f} events per candidate parameter); "
        f"Cox-Snell R2 = {res.r2_cox_snell:.4f}, max = {res.max_r2_cox_snell:.4f}"
    )
    return 0


def _cmd_tables(args: argparse.Namespace) -> int:
    from .io import TableStore

    store = TableStore(args.root, max_depth=args.max_depth)
    frame = store.tables()
    if frame.empty:
        print("No tables found.")
        return 1
    with pd.option_context("display.max_rows", None, "display.max_colwidth", 120, "display.width", 200):
        print(frame.to_string(index=False))
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    from .datasets import FEATURES, make_clinical_risk_data
    from .stability import StabilityAudit

    data = make_clinical_risk_data(n_stays=args.n, random_state=args.seed)
    dev = data[data["site_id"] <= 3]
    ext = data[data["site_id"] > 3]
    result = StabilityAudit(
        "logistic", n_bootstrap=args.bootstraps, bootstrap="cluster", thresholds=(0.1, 0.2),
        random_state=args.seed, n_jobs=args.n_jobs, verbose=1,
    ).run(
        dev[FEATURES], dev["outcome"], groups=dev["patient_id"], subgroups=dev["sex"], ids=dev["stay_id"],
        X_external=ext[FEATURES], y_external=ext["outcome"], subgroups_external=ext["sex"], ids_external=ext["stay_id"],
    )
    paths = result.export(args.out_dir)
    print(result.summary().to_string(index=False))
    print("Outputs written:", ", ".join(paths))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cpmaudit", description="Audit the reliability of individual risk predictions.")
    parser.add_argument("--version", action="version", version=f"cpmaudit {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    st = sub.add_parser("stability", help="bootstrap audit of individual prediction instability")
    st.add_argument("--data", required=True, help="development table (.csv, .csv.gz, .parquet)")
    st.add_argument("--outcome", required=True, help="binary outcome column (0/1)")
    st.add_argument("--features", nargs="*", help="predictor columns (default: all other columns)")
    st.add_argument("--exclude", nargs="*", help="columns to exclude from predictors")
    st.add_argument("--group-col", help="cluster column (e.g. patient id) for the cluster bootstrap")
    st.add_argument("--subgroup-col", help="column with subgroup labels for subgroup summaries")
    st.add_argument("--id-col", help="row identifier carried into individual outputs")
    st.add_argument("--external", help="optional external table with the same columns")
    st.add_argument("--model", default="logistic", help="preset model name (default: logistic)")
    st.add_argument("--bootstrap", default="simple", choices=["simple", "stratified", "cluster"])
    st.add_argument("--bootstraps", type=int, default=200)
    st.add_argument("--threshold", type=float, nargs="+", default=[0.1])
    st.add_argument("--interval-level", type=float, default=0.95)
    st.add_argument("--flip-tolerance", type=float, default=0.1)
    st.add_argument("--n-jobs", type=int, default=1)
    st.add_argument("--seed", type=int, default=2024)
    st.add_argument("--out-dir", default="cpmaudit_output")
    st.add_argument("--no-individual", action="store_true", help="do not write individual-level CSV files")
    st.add_argument("--policy", action="store_true", help="evaluate the default stability policy; exit code 2 on failure")
    st.set_defaults(func=_cmd_stability)

    ss = sub.add_parser("sampsize", help="minimum sample size for a binary-outcome model")
    ss.add_argument("--parameters", type=int, required=True)
    ss.add_argument("--prevalence", type=float, required=True)
    group = ss.add_mutually_exclusive_group(required=True)
    group.add_argument("--cstatistic", type=float)
    group.add_argument("--r2-cs", type=float)
    group.add_argument("--r2-nagelkerke", type=float)
    ss.add_argument("--shrinkage", type=float, default=0.9)
    ss.set_defaults(func=_cmd_sampsize)

    tb = sub.add_parser("tables", help="list tables found under a data directory")
    tb.add_argument("root")
    tb.add_argument("--max-depth", type=int, default=4)
    tb.set_defaults(func=_cmd_tables)

    dm = sub.add_parser("demo", help="run an audit on synthetic data")
    dm.add_argument("--out-dir", default="cpmaudit_demo")
    dm.add_argument("--n", type=int, default=6000)
    dm.add_argument("--bootstraps", type=int, default=100)
    dm.add_argument("--n-jobs", type=int, default=1)
    dm.add_argument("--seed", type=int, default=2024)
    dm.set_defaults(func=_cmd_demo)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
