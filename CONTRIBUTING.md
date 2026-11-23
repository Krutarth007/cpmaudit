# Contributing

Thank you for helping improve cpmaudit.

1. Open an issue describing the bug or proposal (include a minimal example, your Python and package versions).
2. Fork the repository and create a branch.
3. Install the development environment: `pip install -e ".[dev]"`.
4. Add tests for new behaviour in `tests/`; run `pytest` and `ruff check src tests`.
5. Open a pull request describing the change.

Never commit real patient data, even small extracts. Use `cpmaudit.datasets.make_clinical_risk_data` for examples and tests.
