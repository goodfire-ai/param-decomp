# setup
# ONE venv: `param_decomp` carries jax as a normal dependency, so a single `uv sync`
# installs everything into `.venv`. The CPU jax wheel is the base; a GPU host adds the
# `[cuda]` (or `[cuda13]`) extra.
.PHONY: install
install:
	uv sync --no-dev

.PHONY: install-dev
install-dev:
	uv sync
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

# All Python tests live under `param_decomp/tests/`, mirroring the public package.
TEST_PATHS = param_decomp/tests/

# min(16, nproc). XLA already threads within each test, so once the workers saturate the
# box another one buys nothing — the cap only stops a large workstation spawning dozens for
# no gain. testmon is compatible: it ships its own xdist controller/worker sync.
NUM_PROCESSES ?= $(shell (nproc 2>/dev/null || sysctl -n hw.ncpu) | awk '{print ($$1<16?$$1:16)}')

.PHONY: test
test:
	uv run pytest $(TEST_PATHS) --testmon --durations 10 --numprocesses $(NUM_PROCESSES) --dist worksteal

.PHONY: test-all
test-all:
	uv run pytest $(TEST_PATHS) --runslow --durations 10 --numprocesses $(NUM_PROCESSES) --dist worksteal
	$(MAKE) test-multidevice

# CI shards: an exact 3-way partition of `test-all`, one CI job each. One job running
# everything no longer fits the runner: the suite is ~19 min warm (at the 20-min job
# timeout), and a timed-out job is killed before its post steps, so it never saves the
# JAX compile cache and every later run repeats the compile cost. The llama goldens
# split off because they dominate one xdist worker for ~8 min and co-schedule the
# heaviest memory peaks next to the recon end-to-end tests on a 16GB runner. The three
# core integration modules stay on the lab shard: moving their files must not move their
# large memory peaks back beside the rest of the core suite.
LLAMA_GOLDEN_TEST_PATHS = param_decomp/tests/targets/test_llama31.py param_decomp/tests/targets/test_llama_simple_mlp.py
CORE_LAB_TEST_PATHS = \
	param_decomp/tests/core/test_hidden_acts_reconstruction.py \
	param_decomp/tests/core/test_no_checkpointing.py \
	param_decomp/tests/core/test_placed_eval_tiers.py
CORE_TEST_PATHS = param_decomp/tests/core/ param_decomp/tests/targets/
LAB_TEST_PATHS = \
	param_decomp/tests/clustering/ \
	param_decomp/tests/experiments/ \
	param_decomp/tests/infra/ \
	param_decomp/tests/migrations/ \
	param_decomp/tests/vendored_jax/ \
	$(CORE_LAB_TEST_PATHS)

.PHONY: test-ci-llama-goldens
test-ci-llama-goldens:
	uv run pytest $(LLAMA_GOLDEN_TEST_PATHS) --runslow --durations 10 --numprocesses $(NUM_PROCESSES) --dist worksteal

.PHONY: test-ci-core
test-ci-core:
	uv run pytest $(CORE_TEST_PATHS) $(addprefix --ignore=,$(LLAMA_GOLDEN_TEST_PATHS) $(CORE_LAB_TEST_PATHS)) --runslow --durations 10 --numprocesses $(NUM_PROCESSES) --dist worksteal

.PHONY: test-ci-lab-multidevice
test-ci-lab-multidevice:
	uv run pytest $(LAB_TEST_PATHS) --runslow --durations 10 --numprocesses $(NUM_PROCESSES) --dist worksteal
	$(MAKE) test-multidevice

# Tests needing >1 device hang at the default 1, so run them on logical CPU devices.
# Eight is the suite-wide minimum: the faithfulness-fallback (2,2,2) mesh arm needs 8;
# tests wanting exactly a 2 x 2 x 1 topology slice jax.devices() themselves.
MULTIDEVICE_CPU_DEVICE_COUNT = 8

.PHONY: test-multidevice
test-multidevice:
	XLA_FLAGS="--xla_force_host_platform_device_count=$(MULTIDEVICE_CPU_DEVICE_COUNT)" uv run pytest $(TEST_PATHS) -m multidevice --runmultidevice --durations 10 --capture=tee-sys

COVERAGE_DIR=docs/coverage

.PHONY: coverage
coverage:
	uv run pytest $(TEST_PATHS) --cov=param_decomp --runslow
	mkdir -p $(COVERAGE_DIR)
	uv run python -m coverage report -m > $(COVERAGE_DIR)/coverage.txt
	uv run python -m coverage html --directory=$(COVERAGE_DIR)/html/


.PHONY: clean
clean:
	@echo "Cleaning Python cache and build artifacts..."
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .ruff_cache/ .pytest_cache/ .coverage

