# Changelog

## 0.1.0 (2026-11-19)

First public release.

- `StabilityAudit`: bootstrap (simple, stratified, cluster) of the full model-building process; individual MAPE, intervals, classification instability index; subgroup, distribution and external-population stability; calibration and decision-curve instability; Harrell, out-of-bag, .632 and .632+ internal validation; prediction intervals for new individuals; HTML/CSV/JSON export.
- `MultiplicityAudit`: Rashomon set, ambiguity, discrepancy, pairwise disagreement, viable prediction range.
- `instability_curve`: instability as a function of development sample size.
- `minimum_sample_size_binary`: analytic sample-size criteria with C-statistic based R-squared approximation.
- `StabilityPolicy`: pass/fail governance checks.
- `cpmaudit.io.TableStore`: discovery and filtered reading of tables in nested CSV/Parquet layouts; dataset identifier profiles.
- Command-line interface: `stability`, `sampsize`, `tables`, `demo`.
