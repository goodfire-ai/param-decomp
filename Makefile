# setup
.PHONY: install
install:
	uv sync --no-dev

.PHONY: install-lab
install-lab:
	uv sync --all-packages --no-dev

.PHONY: install-dev
install-dev:
	uv sync --all-packages
	uv run pre-commit install

.PHONY: install-all
install-all: install-dev install-app


.PHONY: app
app:
	@uv run --package param-decomp-lab python -m param_decomp_lab.app.run_app

.PHONY: install-app
install-app:
	(cd param_decomp_lab/app/frontend && npm install)

.PHONY: check-app
check-app:
	(cd param_decomp_lab/app/frontend && npm run format && npm run check && npm run lint)

# special install for CI (GitHub Actions) that reduces disk usage and install time
# 1. create a fresh venv with `--clear` -- this is mostly only for local testing of the CI install
# 2. install with `uv sync` but with some special options:
#  > `--frozen` to enforce using the lock file for consistent dependency versions
#  > `--link-mode copy` because:
#    - symlinks/hardlinks dont work half the time anyway
#    - we want to kill the cache after installing before we run the tests
#  > `--extra-index-url` to get cpu-only pytorch wheels. installing with just `uv sync` will download a bunch of cuda stuff we cannot use anyway, since there are no GPUs anyways. takes up a lot of space and makes the install take 5x as long
#  > `--index-strategy unsafe-best-match` because pytorch index won't have every version of each package we need. markupsafe is a particular pain point
# Note: explored the `--compile-bytecode` option for test speedups, nothing came of it. see https://github.com/goodfire-ai/param-decomp/pull/187/commits/740f6a28f4d3378078c917125356b6466f155e71
.PHONY: install-ci
install-ci:
	uv venv --python 3.13 --clear
	uv sync \
		--frozen \
		--all-packages \
		--link-mode copy \
		--extra-index-url https://download.pytorch.org/whl/cpu \
		--index-strategy unsafe-best-match

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

.PHONY: test
test:
	uv run pytest param_decomp/tests/ param_decomp_lab/tests/ --testmon --durations 10

# Use min(4, nproc) for numprocesses. Any more and it slows down the tests.
NUM_PROCESSES ?= $(shell nproc | awk '{print ($$1<4?$$1:4)}')

.PHONY: test-all
test-all:
	uv run pytest param_decomp/tests/ param_decomp_lab/tests/ --runslow --durations 10 --numprocesses $(NUM_PROCESSES) --dist worksteal

COVERAGE_DIR=docs/coverage

.PHONY: coverage
coverage:
	uv run pytest param_decomp/tests/ param_decomp_lab/tests/ --cov=param_decomp --cov=param_decomp_lab --runslow
	mkdir -p $(COVERAGE_DIR)
	uv run python -m coverage report -m > $(COVERAGE_DIR)/coverage.txt
	uv run python -m coverage html --directory=$(COVERAGE_DIR)/html/


.PHONY: clean
clean:
	@echo "Cleaning Python cache and build artifacts..."
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .ruff_cache/ .pytest_cache/ .coverage

