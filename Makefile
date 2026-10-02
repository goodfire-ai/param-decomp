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

# The CI install: a fresh venv (`--clear` matters only when rehearsing this locally) synced
# exactly to the lock.
# Note: explored the `--compile-bytecode` option for test speedups, nothing came of it. see https://github.com/goodfire-ai/param-decomp/pull/187/commits/740f6a28f4d3378078c917125356b6466f155e71
.PHONY: install-ci
install-ci:
	uv venv --python 3.13 --clear
	uv sync --frozen --link-mode copy

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

# What CI enforces: the same three tools as `check`, none of them rewriting the tree.
.PHONY: lint
lint:
	uv run ruff check --no-fix .
	uv run ruff format --check .
	uv run basedpyright

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

# CI shards: an exact partition of `test-all`, one CI job each — runner minutes are cheap,
# wall-clock is not. Groupings are semantic units (a package, a target family, or one
# subsystem's multidevice tests), never a balance of whatever happens to be slow this week.
# The llama goldens stay apart: they dominate one xdist worker and co-schedule the heaviest
# memory peaks next to the recon end-to-end tests. The targets split three ways: the routed
# Qwen3.6 MoE target itself (with the expert code it runs on), the decomposition sites on
# that target (narrow emission, mixer sites, placement), and everything dense. The
# experiment entrypoints and the remaining support suites share one public-library shard.
# The three core integration modules are their own shard: their large memory peaks must
# never sit beside the rest of the core suite. The simulated-multidevice tests run
# single-process by construction, so each subsystem's slice is its own job.
# `param_decomp/tests/infra/test_ci_shards.py` pins the partition: every test file in
# exactly one xdist shard, every multidevice-marked file in exactly one multidevice shard.
LLAMA_GOLDEN_TEST_PATHS = param_decomp/tests/targets/test_llama31.py param_decomp/tests/targets/test_llama_simple_mlp.py
CORE_LAB_TEST_PATHS = \
	param_decomp/tests/core/test_hidden_acts_reconstruction.py \
	param_decomp/tests/core/test_no_checkpointing.py \
	param_decomp/tests/core/test_placed_eval_tiers.py
# The core suite in five: the reconstruction objective with its sources and adversaries;
# the training loop (fit check, compiled training, router-KL integration, performance);
# the remaining loss terms and evaluation; checkpoints and resume; and the state they all
# run on (sources, sharding, the CI function, components). Anything in core/ not listed
# here is state.
CORE_RECON_TEST_PATHS = \
	param_decomp/tests/core/test_capture_comparisons.py \
	param_decomp/tests/core/test_faithfulness_normalization.py \
	param_decomp/tests/core/test_fresh_pgd_c_source_dp_invariance.py \
	param_decomp/tests/core/test_merged_recon.py \
	param_decomp/tests/core/test_persistent_ascent_resume_bias_correction.py \
	param_decomp/tests/core/test_recon_grid_rng.py \
	param_decomp/tests/core/test_recon_log_keys.py \
	param_decomp/tests/core/test_source_grad_mean.py
CORE_TRAINING_TEST_PATHS = \
	param_decomp/tests/core/test_compiled_training.py \
	param_decomp/tests/core/test_fit_check.py \
	param_decomp/tests/core/test_grad_norm_metrics.py \
	param_decomp/tests/core/test_loop_profiling.py \
	param_decomp/tests/core/test_router_kl_integration.py \
	param_decomp/tests/core/test_scheduled_coeffs.py \
	param_decomp/tests/core/test_targeted.py \
	param_decomp/tests/core/test_training_performance.py \
	param_decomp/tests/core/test_training_performance_loop.py \
	param_decomp/tests/core/test_training_scalar_cache.py
CORE_LOSSES_TEST_PATHS = \
	param_decomp/tests/core/test_ci_document_isolation.py \
	param_decomp/tests/core/test_eval_averaging_parity.py \
	param_decomp/tests/core/test_eval_runtime.py \
	param_decomp/tests/core/test_frequency_minimality.py \
	param_decomp/tests/core/test_imp_min.py \
	param_decomp/tests/core/test_imp_min_global_reduction.py \
	param_decomp/tests/core/test_lower_leaky_hard_grad.py \
	param_decomp/tests/core/test_nonlinearity_locality.py \
	param_decomp/tests/core/test_slow_eval.py \
	param_decomp/tests/core/test_uv_norm_ratio.py
CORE_CHECKPOINT_TEST_PATHS = \
	param_decomp/tests/core/test_batch_source_checkpoint.py \
	param_decomp/tests/core/test_checkpoint.py \
	param_decomp/tests/core/test_checkpoint_production_topology.py \
	param_decomp/tests/core/test_checkpointing_config.py \
	param_decomp/tests/core/test_finetune_resume.py
CORE_STATE_TEST_PATHS = param_decomp/tests/core/
TARGETS_MOE_MODEL_TEST_PATHS = \
	param_decomp/tests/routed/ \
	param_decomp/tests/targets/qwen36_moe_hf_parity/ \
	param_decomp/tests/targets/test_dense_masked_qwen36.py \
	param_decomp/tests/targets/test_qwen36_component_activations.py \
	param_decomp/tests/targets/test_qwen36_dtype_census.py \
	param_decomp/tests/targets/test_qwen36_moe.py
TARGETS_MOE_NARROW_TEST_PATHS = param_decomp/tests/targets/test_qwen36_narrow.py
TARGETS_MOE_SITES_TEST_PATHS = \
	param_decomp/tests/targets/test_qwen36_mixer_sites.py \
	param_decomp/tests/targets/test_qwen36_placed_batch_sources.py \
	param_decomp/tests/targets/test_qwen36_placed_ci.py \
	param_decomp/tests/targets/test_qwen36_placed_ci_training.py \
	param_decomp/tests/targets/test_qwen36_pooled_ragged_tp.py \
	param_decomp/tests/targets/test_qwen36_pooled_dense_tp.py \
	param_decomp/tests/targets/test_qwen36_pooled_mixers.py \
	param_decomp/tests/targets/test_qwen36_placement.py
TARGETS_DENSE_TEST_PATHS = param_decomp/tests/targets/
# The LM eval step and its operations are half the lab-lm time; the rest of experiments/lm
# (metric families, run loading, bootstrap) is the other half.
LAB_LM_EVAL_TEST_PATHS = \
	param_decomp/tests/experiments/lm/test_eval.py \
	param_decomp/tests/experiments/lm/test_eval_operations.py
LAB_LM_TEST_PATHS = param_decomp/tests/experiments/lm/
LAB_EXPERIMENTS_TEST_PATHS = \
	param_decomp/tests/experiments/ \
	param_decomp/tests/infra/ \
	param_decomp/tests/migrations/ \
	param_decomp/tests/target_ports/ \
	param_decomp/tests/topology/

# The multidevice slices: the files whose multidevice-marked tests belong to each
# subsystem (their other tests run in the xdist shards above). Placement is the presets
# and their census; worlds is topology invariance — padded stacks at non-dividing node
# counts, and owner/ddp trajectories against a single device.
# Single-process jobs are sized so none runs much past four minutes on a 2-core runner,
# which on the heavier subsystems means one job per file.
MULTIDEVICE_PLACEMENT_TEST_PATHS = \
	param_decomp/tests/target_ports/test_attention_partitioning.py \
	param_decomp/tests/core/test_ci_component_conditioning.py \
	param_decomp/tests/core/test_ci_document_isolation.py \
	param_decomp/tests/core/test_fit_check.py \
	param_decomp/tests/core/test_frequency_state_placement.py \
	param_decomp/tests/core/test_global_transformer_ci.py \
	param_decomp/tests/core/test_placement.py \
	param_decomp/tests/experiments/lm/test_lm_ci_fn_init.py \
	param_decomp/tests/targets/test_document_attention.py \
	param_decomp/tests/targets/test_llama31.py
MULTIDEVICE_PLACED_LLAMA_TEST_PATHS = param_decomp/tests/targets/test_llama_simple_mlp_placed.py
# The Qwen3.6 MoE placement suite is eight single-process jobs, one per contract family:
# each placed full-training program runs 60 to 280 s on a 2-core runner, so a family is
# the largest unit that fits one job's budget, and the three pooled-source programs are
# one job each.
MULTIDEVICE_QWEN_PLACEMENT_TEST_PATHS = param_decomp/tests/targets/test_qwen36_placement.py
MULTIDEVICE_QWEN_CI_FN_TEST_PATHS = param_decomp/tests/targets/test_qwen36_placed_ci.py
MULTIDEVICE_QWEN_CI_FN_TRAINING_TEST_PATHS = param_decomp/tests/targets/test_qwen36_placed_ci_training.py
MULTIDEVICE_QWEN_POOLED_RAGGED_TP_TEST_PATHS = param_decomp/tests/targets/test_qwen36_pooled_ragged_tp.py
MULTIDEVICE_QWEN_POOLED_DENSE_TP_TEST_PATHS = param_decomp/tests/targets/test_qwen36_pooled_dense_tp.py
MULTIDEVICE_QWEN_POOLED_MIXERS_TEST_PATHS = param_decomp/tests/targets/test_qwen36_pooled_mixers.py
MULTIDEVICE_QWEN_BATCH_SOURCES_TEST_PATHS = param_decomp/tests/targets/test_qwen36_placed_batch_sources.py
MULTIDEVICE_QWEN_ROUTER_TEST_PATHS = param_decomp/tests/core/test_router_kl_integration.py
MULTIDEVICE_QWEN_PARITY_TEST_PATHS = \
	param_decomp/tests/core/test_dense_expert_ci.py \
	param_decomp/tests/targets/test_dense_masked_qwen36.py \
	param_decomp/tests/targets/test_qwen36_dtype_census.py
MULTIDEVICE_QWEN_MIXER_SITES_TEST_PATHS = param_decomp/tests/targets/test_qwen36_mixer_sites.py
MULTIDEVICE_PADDING_TEST_PATHS = \
	param_decomp/tests/core/test_stack_padding.py \
	param_decomp/tests/targets/test_stack_padding_placed.py
MULTIDEVICE_REPLICATE_TEST_PATHS = param_decomp/tests/targets/test_step_replicate_invariance.py
MULTIDEVICE_ROUTED_TEST_PATHS = \
	param_decomp/tests/core/test_blocked_factorization.py \
	param_decomp/tests/core/test_blocked_sources.py \
	param_decomp/tests/core/test_selected_ci.py \
	param_decomp/tests/routed/test_dense_experts.py \
	param_decomp/tests/routed/test_expert_source_jobs.py \
	param_decomp/tests/routed/test_experts.py \
	param_decomp/tests/targets/test_gated_delta_chunkwise.py
MULTIDEVICE_EVAL_CONTEXT_TEST_PATHS = param_decomp/tests/experiments/lm/test_eval_context.py
MULTIDEVICE_EVALS_TEST_PATHS = \
	param_decomp/tests/experiments/lm/test_eval_operations.py \
	param_decomp/tests/core/test_placed_eval_tiers.py \
	param_decomp/tests/experiments/lm/test_router_divergence_eval.py \
	param_decomp/tests/experiments/lm/test_well_temperedness.py \
	param_decomp/tests/targets/test_activation_capture.py
MULTIDEVICE_SUBSTRATE_TEST_PATHS = \
	param_decomp/tests/core/test_sigterm_consensus.py \
	param_decomp/tests/core/test_batch_source_checkpoint.py \
	param_decomp/tests/core/test_batch_source_pool.py \
	param_decomp/tests/core/test_checkpoint.py \
	param_decomp/tests/core/test_checkpoint_production_topology.py \
	param_decomp/tests/core/test_masked_forward_remat.py \
	param_decomp/tests/core/test_optim_torch_parity.py \
	param_decomp/tests/core/test_merged_selected_sources.py \
	param_decomp/tests/core/test_source_mask_ingredients.py \
	param_decomp/tests/core/test_source_storage_placement.py \
	param_decomp/tests/core/test_tp_boundary_topology.py \
	param_decomp/tests/experiments/lm/test_checkpoint_resume.py \
	param_decomp/tests/experiments/lm/test_run_inline.py
MULTIDEVICE_SHARDING_TEST_PATHS = param_decomp/tests/core/test_sharding.py

XDIST_FLAGS = --runslow --verbose --durations 10 --numprocesses $(NUM_PROCESSES) --dist worksteal

.PHONY: test-ci-llama-goldens
test-ci-llama-goldens:
	uv run pytest $(LLAMA_GOLDEN_TEST_PATHS) $(XDIST_FLAGS)

.PHONY: test-ci-core-recon
test-ci-core-recon:
	uv run pytest $(CORE_RECON_TEST_PATHS) $(XDIST_FLAGS)

.PHONY: test-ci-core-training
test-ci-core-training:
	uv run pytest $(CORE_TRAINING_TEST_PATHS) $(XDIST_FLAGS)

.PHONY: test-ci-core-losses
test-ci-core-losses:
	uv run pytest $(CORE_LOSSES_TEST_PATHS) $(XDIST_FLAGS)

.PHONY: test-ci-core-checkpoint
test-ci-core-checkpoint:
	uv run pytest $(CORE_CHECKPOINT_TEST_PATHS) $(XDIST_FLAGS)

.PHONY: test-ci-core-integration
test-ci-core-integration:
	uv run pytest $(CORE_LAB_TEST_PATHS) $(XDIST_FLAGS)

.PHONY: test-ci-core-state
test-ci-core-state:
	uv run pytest $(CORE_STATE_TEST_PATHS) $(addprefix --ignore=,$(CORE_LAB_TEST_PATHS) $(CORE_RECON_TEST_PATHS) $(CORE_TRAINING_TEST_PATHS) $(CORE_LOSSES_TEST_PATHS) $(CORE_CHECKPOINT_TEST_PATHS)) $(XDIST_FLAGS)

.PHONY: test-ci-targets-moe-model
test-ci-targets-moe-model:
	uv run pytest $(TARGETS_MOE_MODEL_TEST_PATHS) $(XDIST_FLAGS)

.PHONY: test-ci-targets-moe-narrow
test-ci-targets-moe-narrow:
	uv run pytest $(TARGETS_MOE_NARROW_TEST_PATHS) $(XDIST_FLAGS)

.PHONY: test-ci-targets-moe-sites
test-ci-targets-moe-sites:
	uv run pytest $(TARGETS_MOE_SITES_TEST_PATHS) $(XDIST_FLAGS)

.PHONY: test-ci-targets-dense
test-ci-targets-dense:
	uv run pytest $(TARGETS_DENSE_TEST_PATHS) $(addprefix --ignore=,$(LLAMA_GOLDEN_TEST_PATHS) $(TARGETS_MOE_MODEL_TEST_PATHS) $(TARGETS_MOE_NARROW_TEST_PATHS) $(TARGETS_MOE_SITES_TEST_PATHS)) $(XDIST_FLAGS)

.PHONY: test-ci-lab-lm-eval
test-ci-lab-lm-eval:
	uv run pytest $(LAB_LM_EVAL_TEST_PATHS) $(XDIST_FLAGS)

.PHONY: test-ci-lab-lm
test-ci-lab-lm:
	uv run pytest $(LAB_LM_TEST_PATHS) $(addprefix --ignore=,$(LAB_LM_EVAL_TEST_PATHS)) $(XDIST_FLAGS)

.PHONY: test-ci-lab-experiments
test-ci-lab-experiments:
	uv run pytest $(LAB_EXPERIMENTS_TEST_PATHS) $(addprefix --ignore=,$(LAB_LM_TEST_PATHS)) $(XDIST_FLAGS)

# Tests needing >1 device hang at the default 1, so run them on logical CPU devices.
# Eight is the suite-wide minimum: the faithfulness-fallback (2,2,2) mesh arm needs 8;
# tests wanting exactly a 2 x 2 x 1 topology slice jax.devices() themselves.
MULTIDEVICE_CPU_DEVICE_COUNT = 8
# XLA:CPU sizes its client thread pool from the host's CPU count and issues independent
# collectives in no fixed per-device order. Two collectives over the same device subgroup
# then need a spare executor thread per device to resolve; with one thread per device
# (a 4-vCPU host) they deadlock and the rendezvous watchdog aborts the process. `NPROC`
# is the pool-size override the client honours: two threads per simulated device.
MULTIDEVICE_CLIENT_THREADS = 16
MULTIDEVICE_PYTEST = NPROC=$(MULTIDEVICE_CLIENT_THREADS) XLA_FLAGS="--xla_force_host_platform_device_count=$(MULTIDEVICE_CPU_DEVICE_COUNT)" uv run pytest
MULTIDEVICE_FLAGS = --runslow -m multidevice --runmultidevice --verbose --durations 10 --capture=tee-sys

.PHONY: test-multidevice
test-multidevice:
	$(MULTIDEVICE_PYTEST) $(TEST_PATHS) $(MULTIDEVICE_FLAGS)

.PHONY: test-ci-multidevice-placement
test-ci-multidevice-placement:
	$(MULTIDEVICE_PYTEST) $(MULTIDEVICE_PLACEMENT_TEST_PATHS) $(MULTIDEVICE_FLAGS)

.PHONY: test-ci-multidevice-placed-llama
test-ci-multidevice-placed-llama:
	$(MULTIDEVICE_PYTEST) $(MULTIDEVICE_PLACED_LLAMA_TEST_PATHS) $(MULTIDEVICE_FLAGS)

.PHONY: test-ci-multidevice-padding
test-ci-multidevice-padding:
	$(MULTIDEVICE_PYTEST) $(MULTIDEVICE_PADDING_TEST_PATHS) $(MULTIDEVICE_FLAGS)

.PHONY: test-ci-multidevice-replicate
test-ci-multidevice-replicate:
	$(MULTIDEVICE_PYTEST) $(MULTIDEVICE_REPLICATE_TEST_PATHS) $(MULTIDEVICE_FLAGS)

.PHONY: test-ci-multidevice-routed
test-ci-multidevice-routed:
	$(MULTIDEVICE_PYTEST) $(MULTIDEVICE_ROUTED_TEST_PATHS) $(MULTIDEVICE_FLAGS)

.PHONY: test-ci-multidevice-eval-context
test-ci-multidevice-eval-context:
	$(MULTIDEVICE_PYTEST) $(MULTIDEVICE_EVAL_CONTEXT_TEST_PATHS) $(MULTIDEVICE_FLAGS)

.PHONY: test-ci-multidevice-evals
test-ci-multidevice-evals:
	$(MULTIDEVICE_PYTEST) $(MULTIDEVICE_EVALS_TEST_PATHS) $(MULTIDEVICE_FLAGS)

.PHONY: test-ci-multidevice-sharding
test-ci-multidevice-sharding:
	$(MULTIDEVICE_PYTEST) $(MULTIDEVICE_SHARDING_TEST_PATHS) $(MULTIDEVICE_FLAGS)

.PHONY: test-ci-multidevice-substrate
test-ci-multidevice-substrate:
	$(MULTIDEVICE_PYTEST) $(MULTIDEVICE_SUBSTRATE_TEST_PATHS) $(MULTIDEVICE_FLAGS)

.PHONY: test-ci-multidevice-qwen-placement
test-ci-multidevice-qwen-placement:
	$(MULTIDEVICE_PYTEST) $(MULTIDEVICE_QWEN_PLACEMENT_TEST_PATHS) $(MULTIDEVICE_FLAGS)

.PHONY: test-ci-multidevice-qwen-ci-fn
test-ci-multidevice-qwen-ci-fn:
	$(MULTIDEVICE_PYTEST) $(MULTIDEVICE_QWEN_CI_FN_TEST_PATHS) $(MULTIDEVICE_FLAGS)

.PHONY: test-ci-multidevice-qwen-ci-fn-training
test-ci-multidevice-qwen-ci-fn-training:
	$(MULTIDEVICE_PYTEST) $(MULTIDEVICE_QWEN_CI_FN_TRAINING_TEST_PATHS) $(MULTIDEVICE_FLAGS)

.PHONY: test-ci-multidevice-qwen-pooled-ragged-tp
test-ci-multidevice-qwen-pooled-ragged-tp:
	$(MULTIDEVICE_PYTEST) $(MULTIDEVICE_QWEN_POOLED_RAGGED_TP_TEST_PATHS) $(MULTIDEVICE_FLAGS)

.PHONY: test-ci-multidevice-qwen-pooled-dense-tp
test-ci-multidevice-qwen-pooled-dense-tp:
	$(MULTIDEVICE_PYTEST) $(MULTIDEVICE_QWEN_POOLED_DENSE_TP_TEST_PATHS) $(MULTIDEVICE_FLAGS)

.PHONY: test-ci-multidevice-qwen-pooled-mixers
test-ci-multidevice-qwen-pooled-mixers:
	$(MULTIDEVICE_PYTEST) $(MULTIDEVICE_QWEN_POOLED_MIXERS_TEST_PATHS) $(MULTIDEVICE_FLAGS)

.PHONY: test-ci-multidevice-qwen-batch-sources
test-ci-multidevice-qwen-batch-sources:
	$(MULTIDEVICE_PYTEST) $(MULTIDEVICE_QWEN_BATCH_SOURCES_TEST_PATHS) $(MULTIDEVICE_FLAGS)

.PHONY: test-ci-multidevice-qwen-router
test-ci-multidevice-qwen-router:
	$(MULTIDEVICE_PYTEST) $(MULTIDEVICE_QWEN_ROUTER_TEST_PATHS) $(MULTIDEVICE_FLAGS)

.PHONY: test-ci-multidevice-qwen-parity
test-ci-multidevice-qwen-parity:
	$(MULTIDEVICE_PYTEST) $(MULTIDEVICE_QWEN_PARITY_TEST_PATHS) $(MULTIDEVICE_FLAGS)

.PHONY: test-ci-multidevice-qwen-mixer-sites
test-ci-multidevice-qwen-mixer-sites:
	$(MULTIDEVICE_PYTEST) $(MULTIDEVICE_QWEN_MIXER_SITES_TEST_PATHS) $(MULTIDEVICE_FLAGS)

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
