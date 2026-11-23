"""Input validation and small array helpers shared across the package."""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

EPS = 1e-12


def as_binary_outcome(y: Any, name: str = "y") -> np.ndarray:
    """Return ``y`` as a 1-D ``int8`` array of 0/1 values.

    Accepts lists, numpy arrays, pandas Series and boolean arrays. Raises a
    ``ValueError`` if values other than 0/1 (or True/False) are present or if
    there are missing values.
    """
    if isinstance(y, (pd.Series, pd.Index)):
        arr = y.to_numpy()
    elif isinstance(y, pd.DataFrame):
        if y.shape[1] != 1:
            raise ValueError(f"{name} must be one-dimensional; got a DataFrame with {y.shape[1]} columns.")
        arr = y.iloc[:, 0].to_numpy()
    else:
        arr = np.asarray(y)
    arr = np.ravel(arr)
    if arr.dtype == bool:
        return arr.astype(np.int8)
    if arr.size == 0:
        raise ValueError(f"{name} is empty.")
    try:
        numeric = arr.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain only 0/1 values.") from exc
    if np.isnan(numeric).any():
        raise ValueError(f"{name} contains missing values; remove or impute outcomes before auditing.")
    uniques = np.unique(numeric)
    if not np.all(np.isin(uniques, (0.0, 1.0))):
        raise ValueError(f"{name} must contain only 0/1 values; found {uniques[:10].tolist()}.")
    return numeric.astype(np.int8)


def as_probability(p: Any, name: str = "p", clip: bool = False) -> np.ndarray:
    """Return predicted probabilities as a float64 array, validating the range."""
    arr = np.ravel(np.asarray(p, dtype=float))
    if np.isnan(arr).any():
        raise ValueError(f"{name} contains missing values.")
    if (arr < 0).any() or (arr > 1).any():
        raise ValueError(f"{name} must lie within [0, 1].")
    if clip:
        arr = np.clip(arr, EPS, 1.0 - EPS)
    return arr


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)
    return np.log(p) - np.log1p(-p)


def expit(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def n_rows(X: Any) -> int:
    if hasattr(X, "shape"):
        return int(X.shape[0])
    return len(X)


def take_rows(X: Any, idx: np.ndarray) -> Any:
    """Row-subset ``X`` by integer positions, preserving pandas structure."""
    if isinstance(X, (pd.DataFrame, pd.Series)):
        return X.iloc[idx]
    if hasattr(X, "tocsr"):  # scipy sparse
        return X.tocsr()[idx]
    return np.asarray(X)[idx]


def check_features(X: Any, name: str = "X") -> Any:
    """Light validation of a feature matrix. DataFrames are returned unchanged
    so that column-name based pipelines keep working."""
    if isinstance(X, pd.DataFrame):
        if X.shape[0] == 0:
            raise ValueError(f"{name} has no rows.")
        return X
    if hasattr(X, "tocsr"):
        return X.tocsr()
    arr = np.asarray(X)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional.")
    if arr.shape[0] == 0:
        raise ValueError(f"{name} has no rows.")
    return arr


def check_same_length(*arrays: Any, names: Optional[Sequence[str]] = None) -> None:
    lengths = [n_rows(a) for a in arrays if a is not None]
    if len(set(lengths)) > 1:
        label = ", ".join(names) if names else "inputs"
        raise ValueError(f"{label} must have the same number of rows; got {lengths}.")


def as_label_array(values: Any, n: int, name: str) -> np.ndarray:
    """Return grouping labels (patient ids, subgroup labels, site ids) as an array."""
    if isinstance(values, (pd.Series, pd.Index)):
        arr = values.to_numpy()
    else:
        arr = np.asarray(values)
    arr = np.ravel(arr)
    if arr.shape[0] != n:
        raise ValueError(f"{name} has {arr.shape[0]} entries but the data have {n} rows.")
    if pd.isna(arr).any():
        raise ValueError(f"{name} contains missing values.")
    return arr


def as_thresholds(thresholds: Any) -> tuple:
    if thresholds is None:
        return tuple()
    if np.isscalar(thresholds):
        thresholds = [thresholds]
    out = tuple(float(t) for t in thresholds)
    for t in out:
        if not 0.0 < t < 1.0:
            raise ValueError(f"Decision thresholds must be strictly between 0 and 1; got {t}.")
    return out
