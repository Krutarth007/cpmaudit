# Reproducing the study: individual-level instability of ICU mortality predictions

This folder reproduces every result of the study from the two source databases. It contains the three analysis notebooks, every aggregate table and figure from the published run, the SHA-256 provenance records of that run, and a script that checks the outputs against those records.

**Question.** When an ICU mortality model is rebuilt on a different sample of the same size, how much do individual patients' predicted risks, and the decisions that follow, change? Does that instability carry over, or grow, when the model is used in other hospitals, and can an audit before deployment find the affected patients?

**Design.** Development in MIMIC-IV v2.2 (39,966 adults, first ICU stay, one academic centre, 2008 to 2019); external evaluation in eICU-CRD v2.0 (94,114 adults, 205 US hospitals, 2014 to 2015). Prediction at 24 hours after ICU admission of in-hospital death, from 53 harmonised predictors. Four model classes: logistic regression, ridge logistic regression, random forest and histogram gradient boosting, each refitted on 200 bootstrap samples with the full model-building process repeated every time.

---

## Contents

| Notebook | What it does | Main outputs |
|---|---|---|
| [`01_cohorts_features_quality_v2.ipynb`](01_cohorts_features_quality_v2.ipynb) | Pre-specified design; cohort construction with flow counts; harmonised feature extraction (one value per hour in both databases); plausibility filtering; missingness; Table 1; case-mix comparison; hospital landscape; analytic sample size | `outputs/part1/` |
| [`02_development_instability_v2.ipynb`](02_development_instability_v2.ipynb) | Model development; bootstrap instability audit at Riley and Collins levels 1 to 4; decision instability at 0.10, 0.20 and 0.30; internal validation (Harrell, out-of-bag, .632+); subgroup instability; independent development samples of seven sizes (500 to 16,000) as ground truth; sample size needed for stable individual predictions | `outputs/part2/` |
| [`03_external_multiplicity_governance_v4.ipynb`](03_external_multiplicity_governance_v4.ipynb) | External validation pooled and by hospital (random-effects meta-analysis); instability transport ratio; calibration versus instability by hospital; membership model and missingness-pattern mechanism; deferral strategies; predictive multiplicity; sensitivity analyses; stability policy and model facts; addendum (Monte Carlo precision, effective sample size, threshold sweep, factorial missing-data experiment, three transport directions, extended ground truth, two-way multiverse, simulation study) | `outputs/part3/` |
| [`verify_outputs.py`](verify_outputs.py) | Checks every published output against its SHA-256 provenance record | console report |

Sections of notebook 3 marked as the addendum were added after the first full run; all other analyses, thresholds and the stability policy were fixed in the analysis code before external results were examined.

## 1. Get access to the data

Both databases are free for research but require credentialed access on PhysioNet.

1. Create a PhysioNet account and apply for credentialed access.
2. Complete the required human-subjects research training.
3. Sign the data use agreement for each database:
   - MIMIC-IV v2.2: https://physionet.org/content/mimiciv/2.2/ (https://doi.org/10.13026/6mm1-ek67)
   - eICU-CRD v2.0: https://physionet.org/content/eicu-crd/2.0/ (https://doi.org/10.13026/C2WM1R)
4. Download both releases to local storage.

Patient-level data and anything derived from them must stay on your machine. The repository's `.gitignore` excludes the folders where the notebooks write them (`outputs/part*/data/` and `cache/`).

## 2. Create the environment

The published run used Python 3.11.5 on Windows with the package versions in [`../requirements.txt`](../requirements.txt). The complete list of installed packages is in [`outputs/part3/logs/environment_lock.txt`](outputs/part3/logs/environment_lock.txt).

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt          # run from the repository root
python -m ipykernel install --user --name cpmaudit --display-name "Python (cpmaudit)"
```

The notebooks expect a kernel called `cpmaudit`, which the last command registers. Open them in JupyterLab, Jupyter Notebook or VS Code.

## 3. Tell the notebooks where the data are

Paths and run options are set with environment variables, so the notebooks themselves never need editing.

| Variable | Used by | Default | Meaning |
|---|---|---|---|
| `STUDY_MIMIC_ROOTS` | 1 | built-in local paths | Folder(s) containing MIMIC-IV tables; several folders separated by `;`. The first folder containing a table wins, so a partial Parquet copy can sit in front of the original CSV release |
| `STUDY_EICU_ROOTS` | 1 | built-in local paths | The same for eICU-CRD |
| `STUDY_PROJECT_DIR` | 1, 2, 3 | current folder | Where `outputs/` and `cache/` are written. Run from this `study/` folder to update the published outputs in place |
| `STUDY_VITAL_RULE` | 1 | `first` | Hourly vital-sign rule: `first` valid value in each hour (main analysis) or hourly `mean` (sensitivity analysis) |
| `STUDY_DUCKDB_MEMORY`, `STUDY_DUCKDB_THREADS` | 1 | `4GB`, `4` | Limits for reading large event tables |
| `STUDY_N_JOBS` | 2, 3 | all cores but one | Parallel workers for refits |
| `STUDY_P2_PRESET`, `STUDY_P3_PRESET` | 2, 3 | `full` | `quick` runs a reduced version of every analysis to check the set-up; `full` is the pre-specified analysis. Results are cached separately |
| `STUDY_P3_SENS_MODELS` | 3 | `lr,ridge,rf,hgb` | Model classes included in the sensitivity analyses |
| `STUDY_FORCE_REBUILD` | 1, 2, 3 | `0` | `1` ignores cached intermediate files and recomputes everything |

Example (macOS/Linux):

```bash
export STUDY_MIMIC_ROOTS="/data/mimic-iv-2.2"
export STUDY_EICU_ROOTS="/data/eicu-collaborative-research-database-2.0"
cd study
jupyter lab
```

Example (Windows PowerShell):

```powershell
$env:STUDY_MIMIC_ROOTS = "D:\data\mimic-iv-2.2"
$env:STUDY_EICU_ROOTS  = "D:\data\eicu-collaborative-research-database-2.0"
cd study
jupyter lab
```

`cpmaudit.io.TableStore` finds tables in any layout (flat `.csv.gz` folders, `hosp/` and `icu/` subfolders, Parquet exports, unpacked archives). Converting the large event tables to Parquet once (`TableStore(...).to_parquet(table, folder)`) makes notebook 1 much faster.

## 4. Run the notebooks in order

Run 1, then 2, then 3; each notebook reads the analysis files written by the previous one. To check the set-up first, run notebooks 2 and 3 with `STUDY_P2_PRESET=quick` and `STUDY_P3_PRESET=quick` (minutes rather than hours), then run them again with the default `full` preset.

Recorded step timings of the published run on a 14-core Windows desktop, where some steps reused cached intermediate files: notebook 1 about 2 minutes, notebook 2 about 43 minutes, notebook 3 about 3.8 hours. A first run without caches takes longer, mainly for feature extraction in notebook 1 and the refits in notebook 2. Seeds are fixed in each notebook's configuration and recorded in its provenance manifest.

## 5. Check the outputs

```bash
python verify_outputs.py            # from this folder
python verify_outputs.py --verbose  # list every file
```

For the outputs in this repository the script reports:

```
part1: chain intact; 30 files verified, 0 changed, 0 missing, 6 patient-level files not distributed
part2: chain intact; 38 files verified, 0 changed, 0 missing, 2 patient-level files not distributed
part3: chain intact; 95 files verified, 0 changed, 0 missing, 12 patient-level files not distributed

All published outputs match their provenance records.
```

Each notebook hashes every file it saves with SHA-256 and links the hashes in a chain, so any later change to any output, or to the record itself, is detectable. `outputs/study_manifest.json` holds the final chain hash of each notebook. After a fresh run, the script checks your new outputs against your new records; small numerical differences from the published run can arise with other library versions or hardware.

## What is in `outputs/`

```
outputs/
├── part1/   cohorts, predictors, data quality      (tables/, figures/, logs/)
├── part2/   development and instability            (tables/, figures/, logs/)
├── part3/   external evaluation and governance     (tables/, figures/, logs/)
└── study_manifest.json                             final chain hash per notebook
```

- **Tables** are CSV (and JSON where nested). Start with `part3/tables/results_digest.md` and `results_digest_addendum.md`, which state the key numbers in words.
- **Figures** are PNG at 300 dpi and vector PDF.
- **`part3/tables/table_reproduction_map.csv`** lists every output file with a description and its hash.
- **`part3/tables/table_reporting_checklist.csv`** maps the TRIPOD+AI and TRIPOD-Cluster items to where each is addressed.
- **`part3/tables/model_facts_*.json`** are machine-readable stability summaries for each model, for use in model cards.
- **Logs** hold the provenance manifests and the environment lock file.

**Not included:** analysis datasets, stored refit predictions and caches (`part*/data/`, `cache/`). They contain patient-level information and cannot be shared under the PhysioNet data use agreements; the verification script lists them as not distributed.

## Key results

The short version is in the [main README](../README.md#what-we-found-in-205-hospitals). The full set of results is in the digests and tables named above, with the figures in `outputs/part*/figures/`. Good starting points:

| Figure | File |
|---|---|
| Overview of the main findings | `part3/figures/p3_central_illustration.png` |
| Stable population metrics, unstable individuals | `part2/figures/p2_fig4_population_vs_individual.png` |
| Calibration and instability are different failure modes | `part3/figures/p3_fig4_two_failure_modes.png` |
| Transport follows rare measurement patterns | `part3/figures/p3_fig5b_missingness_mechanism.png` |
| How accurate is the audit against independent-sample truth | `part3/figures/p3_addendum_figA6_extended_ground_truth.png` |
| Stability report card | `part3/figures/p3_fig11_stability_report_card.png` |
