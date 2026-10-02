"""Tests for target-generic evaluation operations bound to LM runs."""

import dataclasses
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from jax import random
from jax.sharding import AxisType, Mesh

from param_decomp.core.adversary import PersistentAdversary, SourcesSgdState
from param_decomp.core.ci_fn.implementations.block_selected.arch import (
    BlockSelectedChunkwiseTransformerCIFn,
)
from param_decomp.core.components import ComponentStacks, SiteSpec
from param_decomp.core.configs import (
    CI_L0Config,
    SgdPGDConfig,
)
from param_decomp.core.init_placed import init_component_stacks_placed, init_sources_sharded
from param_decomp.core.metrics import LineChart
from param_decomp.core.model import CaptureKeys, Positioned
from param_decomp.core.placement import from_config
from param_decomp.core.run import (
    EvalInvocation,
    EvaluationPlan,
    MetricsSink,
    StandaloneOperation,
    StandaloneOperationPlan,
    _run_due_evaluation,
)
from param_decomp.core.schedule import ScheduleConfig
from param_decomp.core.sharding import place_target
from param_decomp.core.train import Decomposition
from param_decomp.experiments.eval_config import EvalConfig
from param_decomp.experiments.lm import eval_operations
from param_decomp.experiments.lm.ci_position_eval import CI_POSITION_CHART_KEY
from param_decomp.experiments.lm.eval_config import (
    CIActiveCountsPerPositionConfig,
    CIMaskedStrategy,
    FreshPGDStrategy,
    PersistentStrategy,
    RouterDivergenceConfig,
    StochasticStrategy,
    WellTemperednessConfig,
)
from param_decomp.experiments.lm.eval_context import LMBatchContext, LMEvalPass
from param_decomp.experiments.lm.input_format import TokenInput, input_sampler
from param_decomp.infra.dataset_store import DatasetMeta, write_dataset_meta
from param_decomp.lm.batch import LMBatch, LMBatchWithDocuments, LMBatchWithRouting
from param_decomp.lm.batch_data import HostTokenBatch, global_token_batch
from param_decomp.targets.qwen36_moe import (
    Qwen36MoeConfig,
    QwenPreparedWeights,
    full_site_cs,
    qwen36_moe_site_specs,
)
from param_decomp.targets.testing import (
    TINY_QWEN36_CS,
    tiny_qwen36_cfg,
    tiny_qwen36_decomposed_model,
    tiny_qwen36_moe_ci_fn_arch,
)
from param_decomp.tests.core.test_eval_runtime import _decomposition, _state


def _eval_config() -> EvalConfig:
    return EvalConfig(
        batch_size=8,
        n_steps=3,
        every=10,
        slow_every=20,
        metrics=[
            WellTemperednessConfig(
                groups=None,
                n_locations=2,
                n_components_per_region=4,
                ablations_per_forward=4,
            )
        ],
    )


def test_global_token_batch_refuses_ids_outside_the_vocab() -> None:
    """The one host boundary every LM token stream crosses: past it the ids are labels
    under jit, where an out-of-range id is at best a NaN CE."""
    from jax.sharding import AxisType, Mesh

    from param_decomp.core.sharding import HSDP_MESH_AXES

    mesh = Mesh(
        np.asarray(jax.devices()[:1]).reshape(1, 1, 1),
        HSDP_MESH_AXES,
        axis_types=(AxisType.Explicit,) * 3,
    )
    tokens = np.array([[0, 5, 31], [7, 1, 2]], dtype=np.int32)
    assert global_token_batch(HostTokenBatch(tokens), mesh, 2, 32).token_ids.shape == (
        2,
        3,
    )
    with pytest.raises(AssertionError, match=r"outside \[0, 31\)"):
        global_token_batch(HostTokenBatch(tokens), mesh, 2, 31)
    with pytest.raises(AssertionError, match="nonnegative"):
        global_token_batch(HostTokenBatch(tokens - 1), mesh, 2, 32)


def test_well_temperedness_receives_run_rng_and_rendering(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def make_operation(
        metric: WellTemperednessConfig,
        schedule: Any,
        model: Any,
        ci_capture_keys: CaptureKeys,
        mesh: Any,
        compiler_options: dict[str, bool | int | str],
        *,
        run_key: Any,
        train_steps: int,
        figure_rendering: Any,
    ) -> StandaloneOperationPlan[Any]:
        captured.update(
            metric=metric,
            model=model,
            ci_capture_keys=ci_capture_keys,
            mesh=mesh,
            compiler_options=compiler_options,
            run_key=run_key,
            train_steps=train_steps,
            figure_rendering=figure_rendering,
        )
        return StandaloneOperationPlan(
            lambda _example: StandaloneOperation(schedule, lambda _context: {})
        )

    renderer = object()
    monkeypatch.setattr(eval_operations, "make_well_temperedness_operation", make_operation)
    monkeypatch.setattr(eval_operations, "BackgroundRenderer", lambda _sink: renderer)
    run_key = jax.random.PRNGKey(11)
    train_steps = 100
    built = SimpleNamespace(
        pd=SimpleNamespace(steps=train_steps, seed=3),
        data=SimpleNamespace(eval_dir=Path("unused")),
        ci_fn=SimpleNamespace(capture_keys=frozenset()),
        target=cast(Any, None),
    )

    batch = LMBatchWithDocuments.from_unsegmented_sequences(jnp.arange(4)[None])
    evaluation = eval_operations.make_lm_evaluation(
        cast(Any, built),
        _eval_config(),
        cast(Any, SimpleNamespace(model=SimpleNamespace(site_names=("site",), sites=()))),
        run_key,
        cast(Any, None),
        n_proc=1,
        sink=cast(Any, SimpleNamespace(accepts_deferred_media=True)),
        compiler_options={},
        sample_batch=lambda _step: batch,
    )
    decomposition = _decomposition()
    prepared = evaluation.prepare(EvalInvocation(decomposition, {}, 30))
    assert len(prepared.operations) == 1
    assert captured["figure_rendering"] is renderer
    assert captured["ci_capture_keys"] == frozenset()
    np.testing.assert_array_equal(captured["run_key"], run_key)
    assert captured["train_steps"] == train_steps


def _write_eval_shards(shards_dir: Path, vocab_size: int, seq_len: int, n_rows: int) -> None:
    shards_dir.mkdir(parents=True)
    write_dataset_meta(
        shards_dir,
        DatasetMeta(
            format_version=2, seq_len=seq_len, tokenizer_name="unused", preprocessing_name="fixture"
        ),
    )
    rows = np.random.default_rng(1).integers(0, vocab_size, size=(n_rows, seq_len), dtype=np.int32)
    pq.write_table(
        pa.table(
            {
                "input_ids": pa.array(rows.tolist(), type=pa.list_(pa.int32())),
            }
        ),
        shards_dir / "shard_00000.parquet",
    )


SEQ_LEN = 8


@dataclasses.dataclass(frozen=True)
class _BoundEvaluation:
    """A tiny Qwen evaluation on an explicit `(data, tp)` mesh and a 16-row corpus."""

    cfg: Qwen36MoeConfig
    sites: tuple[SiteSpec, ...]
    mesh: Mesh
    components: ComponentStacks
    ci_fn: BlockSelectedChunkwiseTransformerCIFn
    evaluation: EvaluationPlan[
        LMBatchWithRouting[LMBatch],
        LMEvalPass[LMBatch, LMBatchWithRouting[LMBatch]],
        LMBatchContext[LMBatch, QwenPreparedWeights, LMBatchWithRouting[LMBatch]],
    ]


def _bind_qwen36_evaluation(
    tmp_path: Path, eval_config: EvalConfig, mesh_shape: tuple[int, int]
) -> _BoundEvaluation:
    cfg = tiny_qwen36_cfg()
    sites = qwen36_moe_site_specs(
        cfg, full_site_cs(cfg, TINY_QWEN36_CS | {"experts_down": 16, "shared_down": 8})
    )
    model = tiny_qwen36_decomposed_model(cfg, sites, random.PRNGKey(0))
    mesh = Mesh(
        np.asarray(jax.devices()[: np.prod(mesh_shape)]).reshape(mesh_shape),
        ("data", "tp"),
        axis_types=(AxisType.Explicit,) * 2,
    )
    rules = from_config("zero1-replicated-resident-moe", mesh, sites)
    placed = place_target(model, rules)
    components = init_component_stacks_placed(sites, random.PRNGKey(1), rules)
    ci_fn = tiny_qwen36_moe_ci_fn_arch(model).initialize(model.sites, rules, random.PRNGKey(2))
    eval_dir = tmp_path / "eval"
    _write_eval_shards(eval_dir, cfg.vocab_size, SEQ_LEN, n_rows=16)
    built = SimpleNamespace(
        pd=SimpleNamespace(steps=20, seed=0),
        data=SimpleNamespace(eval_dir=eval_dir),
        ci_fn=ci_fn,
        target=None,
    )
    with jax.set_mesh(mesh):
        placed_ci_fn = jax.device_put(ci_fn, ci_fn.shardings(mesh))
        evaluation = eval_operations.make_lm_evaluation(
            cast(Any, built),
            eval_config,
            placed,
            random.PRNGKey(3),
            mesh,
            n_proc=1,
            sink=MetricsSink.silent(),
            compiler_options={},
            sample_batch=input_sampler(
                TokenInput(), eval_dir, eval_config.batch_size, 0, mesh, cfg.vocab_size
            ),
        )
    return _BoundEvaluation(
        cfg=cfg,
        sites=sites,
        mesh=mesh,
        components=components,
        ci_fn=placed_ci_fn,
        evaluation=evaluation,
    )


def test_standing_nonlinearity_operation_runs_at_step_zero_on_expert_blocked_sites(
    tmp_path: Path,
) -> None:
    """qwen36's gate/up kinds all carry a `Neurons` partition, so the standing nonlinearity
    operation binds on every qwen36 run with an `eval:` block, and `slow_on_first_step`
    fires it at step 0. The expert kinds' U is block-dim `[E, c, d]`; the operation must
    hand its per-component statistics over in the flat `[C]` order the CI means arrive in."""
    seat = _bind_qwen36_evaluation(
        tmp_path,
        EvalConfig(
            batch_size=2, n_steps=1, every=10, slow_every=10, slow_on_first_step=True, metrics=[]
        ),
        (1, 1),
    )
    with jax.set_mesh(seat.mesh):
        decomposition = Decomposition[LMBatchWithRouting[LMBatch]](
            components=seat.components, ci_fn=seat.ci_fn
        )
        evaluation = seat.evaluation.prepare(EvalInvocation(decomposition, {}, 0))
        record = _run_due_evaluation(evaluation, _state(decomposition, {}), 0)

    assert record is not None
    partitioned = [site for site in seat.sites if site.alignment is not None]
    assert partitioned
    for site in partitioned:
        prefix = f"eval/nonlinearity/sites/{site.name}"
        n_alive = record[f"{prefix}/mean_ci_gt_0/n_components"]
        assert isinstance(n_alive, float) and 0 <= n_alive <= site.C
        soft = record[f"{prefix}/all/soft_use_count_relative_threshold_4"]
        assert isinstance(soft, float) and np.isfinite(soft)
    assert "eval/nonlinearity/aggregates/neuron/all/effective_use_count_per_subcomponent" in record


def test_router_divergence_operation_runs_end_to_end_reading_the_persistent_adversary(
    tmp_path: Path,
) -> None:
    """The bound operation over the pass's real batch contexts: the persistent strategy
    reads the training state's adversary sources off the context, and the pass logs every
    `{kind}/{distance}` key per layer plus its layer mean."""
    strategies = (
        CIMaskedStrategy(),
        StochasticStrategy(),
        FreshPGDStrategy(n_steps=1, step_size=0.1),
        PersistentStrategy(state_key="ppgd"),
    )
    eval_config = EvalConfig(
        batch_size=2,
        n_steps=2,
        every=10,
        slow_every=10,
        slow_on_first_step=False,
        metrics=[RouterDivergenceConfig(strategies=strategies)],
    )
    seat = _bind_qwen36_evaluation(tmp_path, eval_config, (1, 1))
    with jax.set_mesh(seat.mesh):
        adversary = PersistentAdversary(
            sources=init_sources_sharded(
                seat.sites,
                Positioned(n_positions=SEQ_LEN),
                "bc",
                eval_config.batch_size,
                jnp.dtype(jnp.float32),
                random.PRNGKey(7),
                seat.mesh,
            ),
            opt_state=SourcesSgdState(),
            state_key="ppgd",
            optimizer=SgdPGDConfig(lr_schedule=ScheduleConfig.constant(0.1)),
            n_warmup=0,
        )
        decomposition = Decomposition[LMBatchWithRouting[LMBatch]](
            components=seat.components, ci_fn=seat.ci_fn
        )
        evaluation = seat.evaluation.prepare(
            EvalInvocation(
                decomposition,
                {"ppgd": adversary.sources},
                10,
            )
        )
        record = _run_due_evaluation(evaluation, _state(decomposition, {"ppgd": adversary}), 10)

    assert record is not None
    expected = {
        f"eval/router_divergence/{strategy.kind}/{distance}{suffix}"
        for strategy in strategies
        for distance in ("kl", "topk_overlap", "weight_mae")
        for suffix in ("", *(f"/layer_{layer}" for layer in range(seat.cfg.n_layer)))
    }
    assert {key for key in record if key.startswith("eval/router_divergence/")} == expected
    assert all(isinstance(record[key], float) for key in expected)


@pytest.mark.parametrize(
    "mesh_shape",
    [(1, 1), pytest.param((2, 2), marks=pytest.mark.multidevice)],
)
def test_ci_position_chart_is_emitted_only_on_slow_eval_and_averages_to_ci_l0(
    tmp_path: Path,
    mesh_shape: tuple[int, int],
) -> None:
    seat = _bind_qwen36_evaluation(
        tmp_path,
        EvalConfig(
            batch_size=2,
            n_steps=2,
            every=10,
            slow_every=20,
            slow_on_first_step=False,
            metrics=[CIActiveCountsPerPositionConfig(), CI_L0Config(groups=None)],
        ),
        mesh_shape,
    )
    with jax.set_mesh(seat.mesh):
        decomposition = Decomposition[LMBatchWithRouting[LMBatch]](
            components=seat.components, ci_fn=seat.ci_fn
        )
        state = _state(decomposition, {})
        evaluation = seat.evaluation.prepare(EvalInvocation(decomposition, {}, 10))
        fast_only = _run_due_evaluation(evaluation, state, 10)
        record = _run_due_evaluation(evaluation, state, 20)
    assert fast_only is not None and CI_POSITION_CHART_KEY not in fast_only
    assert record is not None
    chart = record[CI_POSITION_CHART_KEY]
    assert isinstance(chart, LineChart)
    np.testing.assert_array_equal(chart.xs, np.arange(SEQ_LEN))
    # The `CI > 0` line is the CI_L0 count with the position axis kept: its position-mean
    # is the sum of the per-site L0 scalars from the same pass.
    (name, active), *_ = chart.series
    assert name == "CI > 0"
    l0_total = sum(
        value
        for key, value in record.items()
        if key.startswith("eval/l0/0.0_") and isinstance(value, float)
    )
    np.testing.assert_allclose(active.mean(), l0_total, rtol=1e-5)
