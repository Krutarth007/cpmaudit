# Methods

This document describes what `cpmaudit` computes and the literature each component follows.

## 1. Notation

Development data: individuals *i = 1..n* with predictors *x_i* and binary outcome *y_i*. The model-building process *M* (preprocessing, selection, tuning and fitting) applied to the full data gives the **original model** with predicted risk *p̂_i*. Applying *M* to bootstrap replicate *b = 1..B* gives a **replicate model** with predicted risk *p̂_bi* for the same individual. An optional external population *j = 1..m* receives *p̂_j* and *p̂_bj*.

## 2. Resampling

* **simple** – *n* rows drawn with replacement.
* **stratified** – rows drawn with replacement separately in events and non-events (fixed number of events).
* **cluster** – clusters (for example patients) drawn with replacement; all rows of a drawn cluster are included. Required when individuals contribute several rows.

Replicates that contain a single outcome class are redrawn. Each replicate has its own seed derived from `numpy.random.SeedSequence(random_state)`, so results are identical with any number of parallel workers.

## 3. Stability at four levels

Following the framework of Riley and Collins (2023):

1. **Mean risk**: mean of *p̂_i* versus the distribution over *b* of the mean of *p̂_bi*.
2. **Risk distribution**: quantiles of *p̂_i* versus the distribution over *b* of the same quantiles of *p̂_bi*.
3. **Subgroups**: levels 4 and decision measures summarised within user-defined subgroups.
4. **Individuals**:
   * individual mean absolute prediction error `MAPE_i = (1/B) Σ_b |p̂_bi − p̂_i|`;
   * **instability index** `= (1/n) Σ_i MAPE_i`;
   * percentile interval `[Q_{α/2}(p̂_·i), Q_{1−α/2}(p̂_·i)]`, its width, SD and maximum absolute change.

## 4. Decision instability

For threshold *t*, the **classification instability index** of individual *i* is

`CII_i(t) = (1/B) Σ_b 1[ 1(p̂_bi ≥ t) ≠ 1(p̂_i ≥ t) ]`,

the share of replicate models whose decision differs from the original model's. Reported summaries are its mean (expected share of changed decisions), the sum (expected number of individuals whose decision changes) and the share of *flip-prone* individuals with `CII_i(t)` above a tolerance (default 0.10).

## 5. Calibration and clinical utility instability

For each replicate model, a smoothed calibration curve is fitted in the population of interest: logistic regression of *y* on a restricted cubic spline of `logit(p̂)` (3–7 knots at standard quantiles; default 4). Curves are evaluated on a grid and never extrapolated beyond the 0.5th–99.5th percentiles of predicted risk. A loess smoother (Austin and Steyerberg, 2019) is available when `statsmodels` is installed.

Summary calibration measures: calibration-in-the-large (intercept with `logit(p̂)` as offset), calibration slope, observed/expected ratio, and the integrated calibration index (ICI, mean absolute difference between predicted and smoothed observed risk) with E50, E90 and Emax.

Net benefit at threshold *t*: `TP/n − FP/n × t/(1−t)` (Vickers and Elkin, 2006), with treat-all and treat-none references.

## 6. Internal validation from the same replicates

* **Harrell's optimism correction**: `optimism = mean_b [ perf(M_b, replicate b) − perf(M_b, original data) ]`; corrected = apparent − optimism.
* **Out-of-bag** performance of each replicate model on rows (clusters) not drawn.
* **.632** estimate: `0.368 × apparent + 0.632 × OOB`.
* **.632+** estimate (Efron and Tibshirani, 1997) for AUROC, Brier score and log loss, using the no-information value of each measure and the relative overfitting rate. Flexible learners (random forests, boosting) can be badly over-optimistic under Harrell's correction; .632+ is usually closer to external performance.

## 7. External populations

Every replicate model is applied to the external population, giving external individual instability (section 3–4) and the distribution of external performance across replicate models: how much a model's measured external performance depends on the particular development sample.

## 8. Predictive multiplicity

Candidate models (different algorithms, hyper-parameters or seeds) are fitted on training data and scored on held-out data. The Rashomon set contains the candidates within ε of the best score (Marx, Calmon and Ustun, 2020). Over the set, with the best model as reference:

* **ambiguity** – share of individuals for whom at least one member's decision differs from the reference;
* **discrepancy** – maximum over members of the share of individuals whose decision differs from the reference;
* **pairwise disagreement** – mean share of disagreeing model pairs;
* **viable prediction range** – per-individual max − min risk across the set (Watson-Daniels, Parkes and Ustun, 2023).

## 9. Sample size

`minimum_sample_size_binary` implements the three criteria for binary outcomes of Riley et al. (2019, 2020):

1. expected shrinkage `S ≥ 0.9`: `n = P / ((S − 1) ln(1 − R²_CS / S))`;
2. difference ≤ δ (0.05) between apparent and adjusted Nagelkerke R²: `S = R²_CS / (R²_CS + δ max R²_CS)`, then as (1);
3. overall risk within ±0.05: `n = (1.96/0.05)² φ(1−φ)`;

with `max R²_CS = 1 − (φ^φ (1−φ)^(1−φ))²`. The anticipated Cox-Snell R² can be approximated from an anticipated C statistic by simulation (Riley, Van Calster and Collins, 2021): the linear predictor is normal with unit variance in events and non-events with mean difference `√2 Φ⁻¹(C)`, a logistic model is fitted and its likelihood-ratio R² is returned.

`instability_curve` complements the analytic criteria empirically: development subsamples of increasing size are audited, and instability is measured on a fixed held-out reference panel so that sizes are comparable.

## 10. Governance

`StabilityPolicy` evaluates user-defined limits (instability index, tail MAPE, share of wide intervals, share of flip-prone individuals per threshold, calibration-slope optimism, minimum replicates). Defaults exist only so the mechanism runs; limits should be set for the decision at hand.

## References

* Austin PC, Steyerberg EW. The Integrated Calibration Index (ICI) and related metrics for quantifying the calibration of logistic regression models. *Statistics in Medicine* 2019;38:4051–4065. doi:10.1002/sim.8281
* Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. *BMJ* 2024;385:e078378.
* Efron B, Tibshirani R. Improvements on cross-validation: the .632+ bootstrap method. *Journal of the American Statistical Association* 1997;92:548–560.
* Harrell FE. *Regression Modeling Strategies*, 2nd ed. Springer, 2015.
* Marx C, Calmon F, Ustun B. Predictive multiplicity in classification. *Proceedings of ICML* 2020.
* Moons KGM, Damen JAA, Kaul T, et al. PROBAST+AI: an updated quality, risk of bias, and applicability assessment tool for prediction models using regression or artificial intelligence methods. *BMJ* 2025;388:e082505.
* Riley RD, Collins GS. Stability of clinical prediction models developed using statistical or machine learning methods. *Biometrical Journal* 2023;65:2200302. doi:10.1002/bimj.202200302
* Riley RD, Pate A, Dhiman P, et al. Clinical prediction models and the multiverse of madness. *BMC Medicine* 2023;21:502.
* Riley RD, Collins GS, Kirton L, et al. Uncertainty of risk estimates from clinical prediction models: rationale, challenges, and approaches. *BMJ* 2025;388:e080749.
* Riley RD, Snell KIE, Ensor J, et al. Minimum sample size for developing a multivariable prediction model: PART II – binary and time-to-event outcomes. *Statistics in Medicine* 2019;38:1276–1296. doi:10.1002/sim.7992
* Riley RD, Ensor J, Snell KIE, et al. Calculating the sample size required for developing a clinical prediction model. *BMJ* 2020;368:m441.
* Riley RD, Van Calster B, Collins GS. A note on estimating the Cox-Snell R² from a reported C statistic (AUROC) to inform sample size calculations for developing a prediction model with a binary outcome. *Statistics in Medicine* 2021;40:859–864. doi:10.1002/sim.8806
* Steyerberg EW, Harrell FE, Borsboom GJ, et al. Internal validation of predictive models: efficiency of some procedures for logistic regression analysis. *Journal of Clinical Epidemiology* 2001;54:774–781.
* Van Calster B, Collins GS, Vickers AJ, et al. Evaluation of performance measures in predictive artificial intelligence models to support medical decisions: overview and guidance. *Lancet Digital Health* 2025;7:100916.
* Van Houwelingen JC, Le Cessie S. Predictive value of statistical models. *Statistics in Medicine* 1990;9:1303–1325.
* Vickers AJ, Elkin EB. Decision curve analysis: a novel method for evaluating prediction models. *Medical Decision Making* 2006;26:565–574.
* Watson-Daniels J, Parkes DC, Ustun B. Predictive multiplicity in probabilistic classification. *Proceedings of AAAI* 2023.
