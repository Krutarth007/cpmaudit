"""Bootstrap resampling schemes.

Three schemes are supported:

* ``"simple"``     - rows sampled with replacement (the classical bootstrap).
* ``"stratified"`` - rows sampled with replacement within outcome strata, so
  every replicate keeps the observed number of events.
* ``"cluster"``    - whole clusters (for example all ICU stays of one patient)
  sampled with replacement. Use this whenever the same patient can contribute
  several rows; otherwise information leaks between replicate and original
  sample and instability is under-estimated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

SCHEMES = ("simple", "stratified", "cluster")


@dataclass
class BootstrapSampler:
    """Draw bootstrap index vectors under a chosen resampling scheme.

    Parameters
    ----------
    scheme:
        One of ``"simple"``, ``"stratified"`` or ``"cluster"``.
    groups:
        Cluster labels, required for ``scheme="cluster"``.
    strata:
        Stratum labels for ``scheme="stratified"``. Defaults to the outcome.
    max_tries:
        Number of redraws allowed when a replicate contains a single outcome
        class (a model cannot be fitted on such a replicate).
    """

    scheme: str = "simple"
    groups: Optional[np.ndarray] = None
    strata: Optional[np.ndarray] = None
    max_tries: int = 50
    _cluster_rows: list = field(default_factory=list, init=False, repr=False)
    _strata_rows: list = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.scheme not in SCHEMES:
            raise ValueError(f"scheme must be one of {SCHEMES}; got {self.scheme!r}.")
        if self.scheme == "cluster":
            if self.groups is None:
                raise ValueError("scheme='cluster' requires group labels (for example patient identifiers).")
            _, inverse = np.unique(np.asarray(self.groups), return_inverse=True)
            order = np.argsort(inverse, kind="stable")
            counts = np.bincount(inverse)
            self._cluster_rows = np.split(order, np.cumsum(counts)[:-1])
        if self.scheme == "stratified" and self.strata is not None:
            self._set_strata(np.asarray(self.strata))

    def _set_strata(self, strata: np.ndarray) -> None:
        self._strata_rows = [np.flatnonzero(strata == s) for s in np.unique(strata)]

    @property
    def n_clusters(self) -> Optional[int]:
        return len(self._cluster_rows) if self.scheme == "cluster" else None

    def draw(self, n: int, rng: np.random.Generator, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Return one vector of row indices (with replacement)."""
        if self.scheme == "stratified" and not self._strata_rows:
            if y is None:
                raise ValueError("Stratified bootstrap needs either strata or the outcome vector.")
            self._set_strata(np.asarray(y))
        for _ in range(self.max_tries):
            idx = self._draw_once(n, rng)
            if y is None or np.unique(y[idx]).size == 2:
                return idx
        raise RuntimeError(
            "Could not draw a bootstrap replicate that contains both outcome classes after "
            f"{self.max_tries} attempts; the number of events is probably too small."
        )

    def _draw_once(self, n: int, rng: np.random.Generator) -> np.ndarray:
        if self.scheme == "simple":
            return rng.integers(0, n, size=n)
        if self.scheme == "stratified":
            parts = [rows[rng.integers(0, rows.size, size=rows.size)] for rows in self._strata_rows]
            return np.concatenate(parts)
        chosen = rng.integers(0, len(self._cluster_rows), size=len(self._cluster_rows))
        return np.concatenate([self._cluster_rows[c] for c in chosen])


def out_of_bag_mask(n: int, idx: np.ndarray) -> np.ndarray:
    """Boolean mask of rows not present in a bootstrap replicate."""
    mask = np.ones(n, dtype=bool)
    mask[np.asarray(idx)] = False
    return mask


def subsample_indices(
    n: int,
    size: int,
    rng: np.random.Generator,
    y: Optional[np.ndarray] = None,
    groups: Optional[np.ndarray] = None,
    stratify: bool = True,
) -> np.ndarray:
    """Draw a subsample *without* replacement of approximately ``size`` rows.

    With ``groups`` the subsample is built from whole clusters (all rows of a
    patient are kept together). With ``stratify`` and no groups, the outcome
    prevalence is preserved.
    """
    if size >= n:
        return np.arange(n)
    if groups is not None:
        uniq, inverse = np.unique(np.asarray(groups), return_inverse=True)
        order = rng.permutation(uniq.size)
        sizes = np.bincount(inverse)[order]
        keep_clusters = order[: int(np.searchsorted(np.cumsum(sizes), size) + 1)]
        return np.flatnonzero(np.isin(inverse, keep_clusters))
    if stratify and y is not None:
        y = np.asarray(y)
        idx = []
        for value in np.unique(y):
            rows = np.flatnonzero(y == value)
            k = int(round(size * rows.size / n))
            idx.append(rng.choice(rows, size=max(k, 1), replace=False))
        return np.sort(np.concatenate(idx))
    return np.sort(rng.choice(n, size=size, replace=False))
