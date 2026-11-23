"""Locate and read tables from local clinical data releases.

Clinical databases are distributed in many physical layouts: flat folders of
``.csv.gz`` files, module sub-folders (``hosp/``, ``icu/``), Parquet exports,
or archives unpacked so that every table sits in a folder of the same name
(``vitalPeriodic.csv/vitalPeriodic.csv``). :class:`TableStore` discovers all of
these automatically so analysis code can refer to tables by name only.

Reading is memory-aware: columns can be selected and rows filtered while the
file is streamed, so a single large event table never has to be loaded whole.
DuckDB is used when installed (fastest, out-of-core); otherwise pandas reads
the file in chunks.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd

_SUFFIXES = (".csv.gz", ".csv", ".tsv.gz", ".tsv", ".parquet", ".pq")
_PRIORITY = {".parquet": 0, ".pq": 0, ".csv.gz": 1, ".tsv.gz": 1, ".csv": 2, ".tsv": 2, "dataset": 0}

FilterValue = Union[Any, Sequence[Any]]


def _split_suffix(name: str):
    low = name.lower()
    for suffix in _SUFFIXES:
        if low.endswith(suffix):
            return name[: -len(suffix)], suffix
    return None, None


def _has_duckdb() -> bool:
    try:
        import duckdb  # noqa: F401

        return True
    except ImportError:
        return False


class TableStore:
    """Catalogue of the tables found under a data directory.

    Parameters
    ----------
    root:
        Top-level directory of the data release.
    max_depth:
        How many directory levels below ``root`` to search.
    duckdb_memory_limit, duckdb_threads:
        Resource limits applied when DuckDB is used (useful on machines with
        little memory, for example ``"3GB"`` and ``2``).

    Examples
    --------
    >>> store = TableStore("/data/eicu-collaborative-research-database-2.0")  # doctest: +SKIP
    >>> store.tables()                                                         # doctest: +SKIP
    >>> vitals = store.read("vitalPeriodic", columns=["patientunitstayid", "observationoffset", "heartrate"],
    ...                     filters={"patientunitstayid": stay_ids})           # doctest: +SKIP
    """

    def __init__(
        self,
        root: Union[str, os.PathLike],
        max_depth: int = 4,
        duckdb_memory_limit: Optional[str] = None,
        duckdb_threads: Optional[int] = None,
    ):
        self.root = Path(root).expanduser()
        if not self.root.is_dir():
            raise FileNotFoundError(f"Data directory not found: {self.root}")
        self.max_depth = int(max_depth)
        self.duckdb_memory_limit = duckdb_memory_limit
        self.duckdb_threads = duckdb_threads
        self._catalogue: Optional[Dict[str, Dict[str, Any]]] = None

    # ----------------------------------------------------------- discovery
    def _scan(self) -> Dict[str, Dict[str, Any]]:
        found: Dict[str, Dict[str, Any]] = {}

        def register(name: str, path: Path, kind: str) -> None:
            rel_parent = path.parent.relative_to(self.root).as_posix() if path.parent != self.root else ""
            if rel_parent.lower().endswith(tuple(_SUFFIXES)):
                rel_parent = "/".join(rel_parent.split("/")[:-1])
            module = rel_parent.split("/")[-1] if rel_parent else ""
            entry = {"name": name, "path": path, "format": kind, "module": module}
            for key in {name.lower(), f"{module}/{name}".lower() if module else name.lower()}:
                current = found.get(key)
                if current is None or _PRIORITY[kind] < _PRIORITY[current["format"]]:
                    found[key] = entry

        base_depth = len(self.root.parts)
        for dirpath, dirnames, filenames in os.walk(self.root):
            here = Path(dirpath)
            if len(here.parts) - base_depth >= self.max_depth:
                dirnames[:] = []
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]
            parquet_files = [f for f in filenames if f.lower().endswith((".parquet", ".pq"))]
            is_dataset = here != self.root and bool(parquet_files) and (
                here.name.lower().endswith((".parquet", ".pq"))
                or all(f.lower().startswith(("part", "data_")) or f.split(".")[0].isdigit() for f in parquet_files)
            )
            if is_dataset:
                dir_stem, _ = _split_suffix(here.name)
                register(dir_stem or here.name, here, "dataset")
            for fname in filenames:
                if fname.startswith("."):
                    continue
                stem, suffix = _split_suffix(fname)
                if stem is None or (is_dataset and suffix in (".parquet", ".pq")):
                    continue
                register(stem, here / fname, suffix)
        return found

    def refresh(self) -> None:
        self._catalogue = None

    @property
    def catalogue(self) -> Dict[str, Dict[str, Any]]:
        if self._catalogue is None:
            self._catalogue = self._scan()
        return self._catalogue

    def tables(self) -> pd.DataFrame:
        """One row per discovered table with module, format, path and size."""
        seen = {}
        for entry in self.catalogue.values():
            seen[str(entry["path"])] = entry
        rows = []
        for entry in seen.values():
            path: Path = entry["path"]
            size = sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) if path.is_dir() else path.stat().st_size
            rows.append({"table": entry["name"], "module": entry["module"], "format": entry["format"],
                         "size_mb": round(size / 1e6, 1), "path": str(path)})
        if not rows:
            return pd.DataFrame(columns=["table", "module", "format", "size_mb", "path"])
        return pd.DataFrame(rows).sort_values(["module", "table"]).reset_index(drop=True)

    def __contains__(self, name: str) -> bool:
        return name.lower() in self.catalogue

    def path(self, name: str) -> Path:
        """Physical path of a table (file or Parquet dataset directory)."""
        entry = self.catalogue.get(name.lower())
        if entry is None:
            close = [e["name"] for k, e in self.catalogue.items() if name.lower() in k][:10]
            hint = f" Similar: {sorted(set(close))}." if close else ""
            raise KeyError(f"Table {name!r} not found under {self.root}.{hint}")
        return entry["path"]

    def format(self, name: str) -> str:
        return self.catalogue[name.lower()]["format"]

    # -------------------------------------------------------------- reading
    def columns(self, name: str) -> List[str]:
        """Column names without reading the data."""
        path, fmt = self.path(name), self.format(name)
        if fmt in (".parquet", ".pq", "dataset"):
            import pyarrow.parquet as pq

            files = sorted(path.rglob("*.parquet")) if path.is_dir() else [path]
            return list(pq.read_schema(files[0]).names)
        sep = "\t" if fmt.startswith(".tsv") else ","
        return list(pd.read_csv(path, nrows=0, sep=sep).columns)

    def read(
        self,
        name: str,
        columns: Optional[Sequence[str]] = None,
        filters: Optional[Mapping[str, FilterValue]] = None,
        nrows: Optional[int] = None,
        engine: str = "auto",
        chunksize: int = 2_000_000,
        dtype: Optional[Mapping[str, Any]] = None,
    ) -> pd.DataFrame:
        """Read a table, optionally selecting columns and filtering rows.

        Parameters
        ----------
        name:
            Table name (case-insensitive), optionally prefixed by its module,
            for example ``"icu/chartevents"``.
        columns:
            Columns to return. Reading fewer columns is the single most
            effective way to reduce memory use.
        filters:
            ``{column: value_or_values}``; rows are kept when the column equals
            the value or is in the list of values. Filters are combined with AND.
        nrows:
            Read at most this many rows (before filtering), for previews.
        engine:
            ``"auto"`` (DuckDB if installed, else pandas), ``"duckdb"`` or ``"pandas"``.
        """
        if engine not in ("auto", "duckdb", "pandas"):
            raise ValueError("engine must be 'auto', 'duckdb' or 'pandas'.")
        use_duckdb = engine == "duckdb" or (engine == "auto" and _has_duckdb() and nrows is None)
        if use_duckdb:
            return self._read_duckdb(name, columns, filters)
        return self._read_pandas(name, columns, filters, nrows, chunksize, dtype)

    def _filter_frame(self, frame: pd.DataFrame, filters: Optional[Mapping[str, FilterValue]]) -> pd.DataFrame:
        if not filters:
            return frame
        mask = np.ones(len(frame), dtype=bool)
        for col, value in filters.items():
            values = list(value) if isinstance(value, (list, tuple, set, np.ndarray, pd.Series, pd.Index)) else [value]
            mask &= frame[col].isin(values).to_numpy()
        return frame.loc[mask]

    def _read_pandas(self, name, columns, filters, nrows, chunksize, dtype) -> pd.DataFrame:
        path, fmt = self.path(name), self.format(name)
        wanted = None
        if columns is not None:
            wanted = list(dict.fromkeys(list(columns) + list(filters or {})))
        if fmt in (".parquet", ".pq", "dataset"):
            frame = pd.read_parquet(path, columns=wanted)
            if nrows is not None:
                frame = frame.head(nrows)
            frame = self._filter_frame(frame, filters)
        else:
            sep = "\t" if fmt.startswith(".tsv") else ","
            reader = pd.read_csv(path, usecols=wanted, sep=sep, chunksize=chunksize, nrows=nrows, dtype=dtype, low_memory=False)
            parts = [self._filter_frame(chunk, filters) for chunk in reader]
            frame = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=wanted)
        if columns is not None:
            frame = frame[list(columns)]
        return frame.reset_index(drop=True)

    def connect(self):
        """A DuckDB connection with resource limits applied."""
        import duckdb

        con = duckdb.connect()
        if self.duckdb_memory_limit:
            con.execute(f"SET memory_limit='{self.duckdb_memory_limit}'")
        if self.duckdb_threads:
            con.execute(f"SET threads={int(self.duckdb_threads)}")
        return con

    def _source_literal(self, name: str) -> str:
        source, params = self._source_sql(name)
        return source.replace("?", "'" + params[0].replace("'", "''") + "'", 1)

    def _source_sql(self, name: str) -> tuple:
        path, fmt = self.path(name), self.format(name)
        if fmt == "dataset":
            return "read_parquet(?)", [str(path / "**" / "*.parquet")]
        if fmt in (".parquet", ".pq"):
            return "read_parquet(?)", [str(path)]
        return "read_csv_auto(?, header=true, sample_size=-1)", [str(path)]

    def _read_duckdb(self, name, columns, filters) -> pd.DataFrame:
        con = self.connect()
        try:
            source, params = self._source_sql(name)
            select = "*" if columns is None else ", ".join(_quote(c) for c in columns)
            where = []
            for i, (col, value) in enumerate((filters or {}).items()):
                values = list(value) if isinstance(value, (list, tuple, set, np.ndarray, pd.Series, pd.Index)) else [value]
                tmp = f"__filter_{i}"
                con.register(tmp, pd.DataFrame({"v": values}))
                where.append(f"{_quote(col)} IN (SELECT v FROM {tmp})")
            sql = f"SELECT {select} FROM {source}"
            if where:
                sql += " WHERE " + " AND ".join(where)
            return con.execute(sql, params).df()
        finally:
            con.close()

    def query(self, sql: str, tables: Optional[Iterable[str]] = None) -> pd.DataFrame:
        """Run SQL with DuckDB, exposing each requested table as a view of the
        same name (all tables if ``tables`` is None)."""
        con = self.connect()
        try:
            names = list(tables) if tables is not None else sorted({e["name"] for e in self.catalogue.values()})
            for name in names:
                con.execute(f"CREATE OR REPLACE VIEW {_quote(name)} AS SELECT * FROM {self._source_literal(name)}")
            return con.execute(sql).df()
        finally:
            con.close()

    def to_parquet(self, name: str, out_dir: Union[str, os.PathLike], overwrite: bool = False) -> Path:
        """Convert one table to Parquet (DuckDB streaming conversion if
        available, otherwise pyarrow's streaming CSV reader)."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        target = out_dir / f"{self.catalogue[name.lower()]['name']}.parquet"
        if target.exists() and not overwrite:
            return target
        if _has_duckdb():
            con = self.connect()
            try:
                dest = str(target).replace("'", "''")
                con.execute(f"COPY (SELECT * FROM {self._source_literal(name)}) TO '{dest}' (FORMAT PARQUET)")
            finally:
                con.close()
            return target
        try:
            import pyarrow.csv as pacsv
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise ImportError("Parquet conversion needs duckdb or pyarrow: `pip install cpmaudit[data]`.") from exc
        reader = pacsv.open_csv(str(self.path(name)), read_options=pacsv.ReadOptions(block_size=64 << 20))
        writer = None
        try:
            for batch in reader:
                if writer is None:
                    writer = pq.ParquetWriter(str(target), batch.schema)
                writer.write_batch(batch)
        finally:
            if writer is not None:
                writer.close()
        return target


def _quote(identifier: str) -> str:
    return '"' + str(identifier).replace('"', '""') + '"'


def resolve_table(root: Union[str, os.PathLike], name: str) -> Path:
    """Shortcut: path of table ``name`` anywhere under ``root``."""
    return TableStore(root).path(name)
