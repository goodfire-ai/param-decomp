# setup
# ONE venv for the whole workspace: the JAX trainer core (`param_decomp` + `pretrain` +
# `vendored_jax`) is the root distribution and carries jax as a normal dependency, so a
# single `uv sync --all-packages` installs core + config + lab into one `.venv`. The CPU
# jax wheel is the base; the CUDA wheel is the `[cuda]` extra the per-run launch workspace
# installs.
.PHONY: install
install:
	uv sync --no-dev

.PHONY: install-lab
install-lab:
	uv sync --all-packages --no-dev

.PHONY: install-dev
install-dev:
	uv sync --all-packages
	uv run --no-sync pre-commit install

# special install for CI (GitHub Actions) that reduces disk usage and install time
# 1. create a fresh venv with `--clear` -- this is mostly only for local testing of the CI install
# 2. install with `uv sync` but with some special options:
#  > `--frozen` to enforce using the lock file for consistent dependency versions
#  > `--link-mode copy` because symlinks/hardlinks dont work half the time anyway
# Note: explored the `--compile-bytecode` option for test speedups, nothing came of it. see https://github.com/goodfire-ai/param-decomp/pull/187/commits/740f6a28f4d3378078c917125356b6466f155e71
.PHONY: install-ci
install-ci:
	uv venv --python 3.13 --clear
	uv sync \
		--frozen \
		--all-packages \
		--link-mode copy

# checks
.PHONY: type
type:
	uv run basedpyright

.PHONY: format
format:
	# Fix all autofixable problems (which sorts imports) then format errors
	uv run ruff check --fix
	uv run ruff format

.PHONY: check
check: format type

.PHONY: check-pre-commit
check-pre-commit:
	SKIP=no-commit-to-branch pre-commit run -a --hook-stage commit

# tests

# `param_decomp/tests/` is the JAX trainer core suite (incl. the LM equivalence goldens);
# `param_decomp_lab/{tests,experiments}/` the lab suites (the toy TMS/ResidMLP tests live
# beside their models under experiments/).
TEST_PATHS = param_decomp/tests/ param_decomp_lab/tests/ param_decomp_lab/experiments/

.PHONY: test
test:
	uv run pytest $(TEST_PATHS) --testmon --durations 10

# Use min(4, nproc) for numprocesses. Any more and it slows down the tests.
NUM_PROCESSES ?= $(shell nproc | awk '{print ($$1<4?$$1:4)}')

.PHONY: test-all
test-all:
	uv run pytest $(TEST_PATHS) --runslow --durations 10 --numprocesses $(NUM_PROCESSES) --dist worksteal
	$(MAKE) test-multidevice

# Tests needing >1 device (sharding / checkpoint topology). They hang at the default 1
# device, so they're skipped in the 1-device passes and run here under SIMULATED CPU
# devices (XLA_FLAGS). `make test-all` runs this automatically as a second pass; invoke it
# directly only to run the subset alone (e.g. iterating on sharding/checkpoint).
.PHONY: test-multidevice
test-multidevice:
	XLA_FLAGS="--xla_force_host_platform_device_count=4" uv run pytest $(TEST_PATHS) -m multidevice --runmultidevice --durations 10

COVERAGE_DIR=docs/coverage

.PHONY: coverage
coverage:
	uv run pytest $(TEST_PATHS) --cov=param_decomp --cov=param_decomp_lab --runslow
	mkdir -p $(COVERAGE_DIR)
	uv run python -m coverage report -m > $(COVERAGE_DIR)/coverage.txt
	uv run python -m coverage html --directory=$(COVERAGE_DIR)/html/


.PHONY: clean
clean:
	@echo "Cleaning Python cache and build artifacts..."
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .ruff_cache/ .pytest_cache/ .coverage

