import numpy as np
import pandas as pd
import pytest

from cpmaudit.cli import main
from cpmaudit.io import EICU_CRD, MIMIC_IV, TableStore, get_profile


@pytest.fixture
def nested(tmp_path):
    root = tmp_path / "release"
    (root / "vitalPeriodic.csv").mkdir(parents=True)
    vp = pd.DataFrame({"patientunitstayid": np.repeat([1, 2, 3], 3), "heartrate": np.arange(9) + 60.0})
    vp.to_csv(root / "vitalPeriodic.csv" / "vitalPeriodic.csv", index=False)
    (root / "hosp").mkdir()
    pd.DataFrame({"subject_id": [1, 2], "hadm_id": [5, 6]}).to_csv(root / "hosp" / "admissions.csv.gz", index=False)
    return root


def test_discovery_and_reading(nested):
    store = TableStore(nested)
    tables = store.tables()
    assert set(tables["table"]) == {"vitalPeriodic", "admissions"}
    assert "VITALPERIODIC" in store and "hosp/admissions" in store
    out = store.read("vitalperiodic", columns=["heartrate"], filters={"patientunitstayid": [1, 3]}, engine="pandas")
    assert out["heartrate"].tolist() == [60.0, 61.0, 62.0, 66.0, 67.0, 68.0]
    assert store.columns("admissions") == ["subject_id", "hadm_id"]
    with pytest.raises(KeyError):
        store.path("labevents")


def test_duckdb_engine(nested, tmp_path):
    pytest.importorskip("duckdb")
    store = TableStore(nested, duckdb_threads=1)
    out = store.read("vitalPeriodic", filters={"patientunitstayid": 2}, engine="duckdb")
    assert len(out) == 3
    agg = store.query('SELECT patientunitstayid, max(heartrate) AS hr FROM "vitalPeriodic" GROUP BY 1 ORDER BY 1', tables=["vitalPeriodic"])
    assert agg["hr"].tolist() == [62.0, 65.0, 68.0]
    target = store.to_parquet("vitalPeriodic", tmp_path / "pq")
    assert target.exists()


def test_profiles():
    assert get_profile("mimic4") is MIMIC_IV
    assert get_profile("eICU") is EICU_CRD
    assert MIMIC_IV.module_of("chartevents") == "icu"
    assert EICU_CRD.cluster_column() == "uniquepid"
    with pytest.raises(KeyError):
        get_profile("unknown")


def test_cli_sampsize_tables_demo(nested, tmp_path, capsys):
    assert main(["sampsize", "--parameters", "24", "--prevalence", "0.139", "--r2-cs", "0.051"]) == 0
    assert "4115" in capsys.readouterr().out
    assert main(["tables", str(nested)]) == 0
    assert main(["demo", "--out-dir", str(tmp_path / "demo"), "--n", "1200", "--bootstraps", "5"]) == 0
    assert (tmp_path / "demo" / "report.html").exists()


def test_cli_stability(tmp_path, data):
    path = tmp_path / "cohort.csv"
    data.drop(columns=["true_risk"]).to_csv(path, index=False)
    code = main([
        "stability", "--data", str(path), "--outcome", "outcome", "--exclude", "stay_id", "site_id",
        "--group-col", "patient_id", "--bootstrap", "cluster", "--bootstraps", "5", "--out-dir", str(tmp_path / "o"),
    ])
    assert code == 0 and (tmp_path / "o" / "summary.csv").exists()
