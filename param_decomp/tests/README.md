# Tests

The Python test tree mirrors the public package:

- `core/`, `targets/`, `experiments/`, `infra/`, `migrations/`, `clustering/`, and
  `vendored_jax/` test the corresponding source packages.
- Fixtures and golden generators stay beside the tests that consume their artifacts.
- The repository-root `conftest.py` applies the shared Pytest hooks.

Run the full Python suite from the repository root with `make test-all`.
