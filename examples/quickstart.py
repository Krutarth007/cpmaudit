"""Minimal end-to-end example on synthetic data (no credentials needed).

Run:  python examples/quickstart.py
"""

from cpmaudit import StabilityAudit, StabilityPolicy, minimum_sample_size_binary
from cpmaudit.datasets import FEATURES, make_clinical_risk_data


def main() -> None:
    data = make_clinical_risk_data(n_stays=6000, random_state=1)
    dev = data[data["site_id"] <= 3]
    ext = data[data["site_id"] > 3]

    audit = StabilityAudit(
        model="logistic",
        n_bootstrap=200,
        bootstrap="cluster",
        thresholds=(0.1, 0.2),
        random_state=2024,
        keep_models=True,
        n_jobs=-1,
    )
    result = audit.run(
        dev[FEATURES],
        dev["outcome"],
        groups=dev["patient_id"],
        subgroups=dev["sex"],
        ids=dev["stay_id"],
        X_external=ext[FEATURES],
        y_external=ext["outcome"],
        subgroups_external=ext["sex"],
    )

    print(result)
    print(result.summary().to_string(index=False))
    print(result.optimism()[["metric", "apparent", "optimism_corrected", "estimate_632plus"]].to_string(index=False))

    new_patients = ext[FEATURES].head(5)
    print(result.predict_interval(new_patients))

    policy = StabilityPolicy(max_share_flip_prone=0.10)
    print(policy.evaluate(result))

    print(minimum_sample_size_binary(n_parameters=14, prevalence=0.12, c_statistic=0.78).criteria)

    paths = result.export("quickstart_output")
    print("Report:", paths.get("report.html"))


if __name__ == "__main__":
    main()
