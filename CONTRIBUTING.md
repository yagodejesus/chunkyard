# Contributing

Issues and pull requests are welcome.

1. Fork the repository and create a branch.
2. `pip install -e ".[dev]"`
3. Add tests in `tests/` for your change.
4. Run `pytest && ruff check .` before opening the pull request.

Please use synthetic data in examples and issues — never real secrets, customer data or billing exports.
