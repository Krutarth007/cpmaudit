"""Model specifications accepted by the audits.

An audit needs to *repeat the complete model-building process* on every
bootstrap replicate: imputation, scaling, variable selection, tuning and
fitting. Three kinds of specification are accepted:

1. A preset name such as ``"logistic"`` or ``"hist_gradient_boosting"``.
2. Any scikit-learn compatible estimator, including a full ``Pipeline`` or a
   ``GridSearchCV`` object. It is cloned before every fit so that tuning and
   preprocessing are repeated inside each replicate.
3. A callable ``build(X, y, random_state=...)`` returning a fitted object that
   exposes ``predict_proba`` or is itself callable on new data and returns
   risks. Use this for statistical models or custom code.
"""

from __future__ import annotations

import inspect
import warnings
from typing import Any, Callable, List, Optional, Sequence

import numpy as np
import pandas as pd

from ._validation import expit

PRESETS = (
    "logistic",
    "logistic_l2",
    "logistic_l1",
    "random_forest",
    "hist_gradient_boosting",
    "lightgbm",
    "xgboost",
)


def _sklearn_version() -> tuple:
    import sklearn

    parts = []
    for piece in sklearn.__version__.split(".")[:2]:
        digits = "".join(ch for ch in piece if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


def _one_hot_encoder():
    from sklearn.preprocessing import OneHotEncoder

    if _sklearn_version() >= (1, 2):
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    return OneHotEncoder(handle_unknown="ignore", sparse=False)  # pragma: no cover


def _logistic(kind: str):
    from sklearn.linear_model import LogisticRegression

    if kind == "logistic":
        return LogisticRegression(C=np.inf, max_iter=5000)
    if kind == "logistic_l2":
        return LogisticRegression(C=1.0, max_iter=5000)
    if _sklearn_version() >= (1, 8):
        return LogisticRegression(C=1.0, l1_ratio=1.0, solver="saga", max_iter=5000)
    return LogisticRegression(C=1.0, penalty="l1", solver="saga", max_iter=5000)  # pragma: no cover


def make_model(name: str, categorical_features: Optional[Sequence[str]] = None, **params: Any):
    """Build a ready-to-fit scikit-learn pipeline from a preset name.

    Parameters
    ----------
    name:
        One of ``"logistic"`` (unpenalised), ``"logistic_l2"``, ``"logistic_l1"``,
        ``"random_forest"``, ``"hist_gradient_boosting"``, ``"lightgbm"`` or
        ``"xgboost"`` (the last two need the optional packages).
    categorical_features:
        Column names to one-hot encode. Remaining columns are treated as
        numeric, median-imputed with missingness indicators.
    **params:
        Overrides passed to the final estimator via ``set_params``.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    if name not in PRESETS:
        raise ValueError(f"Unknown preset {name!r}. Choose one of {PRESETS}.")

    linear = name.startswith("logistic")
    if name in ("logistic", "logistic_l2", "logistic_l1"):
        estimator = _logistic(name)
    elif name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier

        estimator = RandomForestClassifier(n_estimators=300, min_samples_leaf=5, max_features="sqrt", n_jobs=1)
    elif name == "hist_gradient_boosting":
        from sklearn.ensemble import HistGradientBoostingClassifier

        estimator = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, early_stopping=False)
    elif name == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
        except ImportError as exc:
            raise ImportError("The 'lightgbm' preset requires `pip install lightgbm`.") from exc
        estimator = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, n_jobs=1, verbose=-1)
    else:
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError("The 'xgboost' preset requires `pip install xgboost`.") from exc
        estimator = XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=4, n_jobs=1, tree_method="hist", eval_metric="logloss"
        )
    if params:
        estimator.set_params(**params)

    numeric_steps: List = [("impute", SimpleImputer(strategy="median", add_indicator=True))]
    if linear:
        numeric_steps.append(("scale", StandardScaler()))
    if name == "hist_gradient_boosting" and not categorical_features:
        numeric_steps = []  # native missing-value support

    if categorical_features:
        categorical = Pipeline(
            [("impute", SimpleImputer(strategy="most_frequent")), ("encode", _one_hot_encoder())]
        )
        numeric = Pipeline(numeric_steps) if numeric_steps else "passthrough"
        pre = ColumnTransformer(
            [("categorical", categorical, list(categorical_features))],
            remainder=numeric,
        )
        return Pipeline([("preprocess", pre), ("model", estimator)])
    if not numeric_steps:
        return Pipeline([("model", estimator)])
    return Pipeline(numeric_steps + [("model", estimator)])


def _categorical_columns(X: Any) -> List[str]:
    if not isinstance(X, pd.DataFrame):
        return []
    cols = []
    for col, dtype in X.dtypes.items():
        if (
            pd.api.types.is_object_dtype(dtype)
            or isinstance(dtype, pd.CategoricalDtype)
            or pd.api.types.is_bool_dtype(dtype)
            or pd.api.types.is_string_dtype(dtype)
        ):
            cols.append(col)
    return cols


class ModelBuilder:
    """Uniform interface around presets, estimators and callables.

    Parameters
    ----------
    model:
        Preset name, scikit-learn compatible estimator or callable (see the
        module docstring).
    predict_fn:
        Optional ``predict_fn(fitted_model, X) -> risks``. Needed only for
        objects that do not expose ``predict_proba``.
    reseed:
        If True (default), every ``random_state`` parameter of an estimator is
        set to the replicate seed so that stochastic learners vary between
        replicates in a reproducible way.
    """

    def __init__(self, model: Any, predict_fn: Optional[Callable] = None, reseed: bool = True):
        if isinstance(model, ModelBuilder):
            self.model, self.predict_fn, self.reseed = model.model, model.predict_fn, model.reseed
            return
        self.model = model
        self.predict_fn = predict_fn
        self.reseed = reseed
        self._preset_template = None
        if isinstance(model, str):
            if model not in PRESETS:
                raise ValueError(f"Unknown preset {model!r}. Choose one of {PRESETS}.")
        elif not (hasattr(model, "fit") or callable(model)):
            raise TypeError("model must be a preset name, an estimator with .fit, or a callable.")

    @property
    def description(self) -> str:
        if isinstance(self.model, str):
            return self.model
        if hasattr(self.model, "fit"):
            return type(self.model).__name__
        return getattr(self.model, "__name__", "callable")

    def _template(self, X: Any):
        if isinstance(self.model, str):
            if self._preset_template is None:
                self._preset_template = make_model(self.model, categorical_features=_categorical_columns(X) or None)
            return self._preset_template
        return self.model

    def fit(self, X: Any, y: np.ndarray, random_state: Optional[int] = None) -> Any:
        template = self._template(X)
        if hasattr(template, "fit") and hasattr(template, "get_params"):
            from sklearn.base import clone

            est = clone(template)
            if self.reseed and random_state is not None:
                seeds = {k: int(random_state) for k in est.get_params(deep=True) if k.endswith("random_state")}
                if seeds:
                    est.set_params(**seeds)
            return est.fit(X, y)
        if hasattr(template, "fit"):
            import copy

            est = copy.deepcopy(template)
            return est.fit(X, y)
        kwargs = {}
        try:
            if "random_state" in inspect.signature(template).parameters:
                kwargs["random_state"] = random_state
        except (TypeError, ValueError):  # pragma: no cover - builtins without signature
            pass
        return template(X, y, **kwargs)

    def predict_risk(self, fitted: Any, X: Any) -> np.ndarray:
        if self.predict_fn is not None:
            risk = self.predict_fn(fitted, X)
        elif hasattr(fitted, "predict_proba"):
            proba = np.asarray(fitted.predict_proba(X))
            classes = list(getattr(fitted, "classes_", [0, 1]))
            col = classes.index(1) if 1 in classes else proba.shape[1] - 1
            risk = proba[:, col]
        elif callable(fitted):
            risk = fitted(X)
        elif hasattr(fitted, "decision_function"):
            risk = expit(np.asarray(fitted.decision_function(X), dtype=float))
        else:
            raise TypeError(
                "Cannot obtain risks from the fitted model; supply predict_fn(model, X) returning probabilities."
            )
        risk = np.ravel(np.asarray(risk, dtype=float))
        if np.isnan(risk).any():
            raise ValueError("The model returned missing predicted risks.")
        return np.clip(risk, 0.0, 1.0)


def fit_quietly(builder: ModelBuilder, X: Any, y: np.ndarray, random_state: Optional[int]):
    """Fit while capturing convergence warnings; returns ``(model, n_warnings)``."""
    from sklearn.exceptions import ConvergenceWarning

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        warnings.simplefilter("ignore", FutureWarning)
        warnings.simplefilter("ignore", RuntimeWarning)
        model = builder.fit(X, y, random_state=random_state)
    n_conv = sum(1 for w in caught if issubclass(w.category, ConvergenceWarning))
    return model, n_conv
