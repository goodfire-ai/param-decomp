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

# `param_decomp/tests/` is the JAX trainer core suite; `param_decomp_targets/tests/` the
# per-target parity/golden suites (incl. the LM equivalence goldens);
# `param_decomp_lab/{tests,experiments}/` the lab suites (the toy TMS/ResidMLP experiment
# tests live beside their composition roots under experiments/).
TEST_PATHS = param_decomp/core/tests/ param_decomp/targets/tests/ param_decomp_lab/tests/ param_decomp_lab/experiments/

.PHONY: test
test:
	uv run pytest $(TEST_PATHS) --testmon --durations 10

# Use min(4, nproc) for numprocesses. Any more and it slows down the tests.
NUM_PROCESSES ?= $(shell nproc | awk '{print ($$1<4?$$1:4)}')

.PHONY: test-all
test-all:
	uv run pytest $(TEST_PATHS) --runslow --durations 10 --numprocesses $(NUM_PROCESSES) --dist worksteal
	$(MAKE) test-multidevice

# CI shards: an exact 3-way partition of `test-all`, one CI job each. One job running
# everything no longer fits the runner: the suite is ~19 min warm (at the 20-min job
# timeout), and a timed-out job is killed before its post steps, so it never saves the
# JAX compile cache and every later run repeats the compile cost. The llama goldens
# split off because they dominate one xdist worker for ~8 min and co-schedule the
# heaviest memory peaks next to the recon end-to-end tests on a 16GB runner.
LLAMA_GOLDEN_TEST_PATHS = param_decomp/targets/tests/test_llama8b.py param_decomp/targets/tests/test_llama_simple_mlp.py

.PHONY: test-ci-llama-goldens
test-ci-llama-goldens:
	uv run pytest $(LLAMA_GOLDEN_TEST_PATHS) --runslow --durations 10 --numprocesses $(NUM_PROCESSES) --dist worksteal

.PHONY: test-ci-core
test-ci-core:
	uv run pytest param_decomp/core/tests/ param_decomp/targets/tests/ $(addprefix --ignore=,$(LLAMA_GOLDEN_TEST_PATHS)) --runslow --durations 10 --numprocesses $(NUM_PROCESSES) --dist worksteal

.PHONY: test-ci-lab-multidevice
test-ci-lab-multidevice:
	uv run pytest param_decomp_lab/tests/ param_decomp_lab/experiments/ --runslow --durations 10 --numprocesses $(NUM_PROCESSES) --dist worksteal
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

