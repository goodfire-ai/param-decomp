"""The LM eval pass's shared batch context on either model-output edge, and the masked-
forward operations that read it, with data and expert parallelism.

The context step pins the clean output over the batch axes. On the streamed edge that
output is the factored `StreamedLinearOutput` package, not an array — a raw array pin
died at trace, so the first eval tick of a streamed-edge run never ran. The context must
build, and the CE/KL scorer must consume it, on both edges of the tiny qwen36_moe target
placed on an explicit `(data, tp)` mesh; the router-divergence step must run every
strategy over it (its fresh-PGD ascent scores the output edge, its persistent strategy
reads sharded sources)."""

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import random
from jax.sharding import AxisType, Mesh

from param_decomp.core.ci_fn.interface import CIFn
from param_decomp.core.components import SelectedCI
from param_decomp.core.init_placed import init_component_stacks_placed, init_sources_sharded
from param_decomp.core.model import EMPTY_CAPTURE_KEYS, CaptureKeys, Positioned
from param_decomp.core.placement import from_config
from param_decomp.core.sharding import place_target, shard_batch
from param_decomp.experiments.lm.eval import make_ce_kl_scorer
from param_decomp.experiments.lm.eval_config import (
    CIMaskedStrategy,
    FreshPGDStrategy,
    PersistentStrategy,
    RouterDivergenceConfig,
    StochasticStrategy,
)
from param_decomp.experiments.lm.eval_context import (
    LMBatchContext,
    make_lm_batch_context_step,
    prepared_batch_from_context,
)
from param_decomp.experiments.lm.load_run import PlacedQwen
from param_decomp.experiments.lm.router_divergence_eval import (
    make_router_divergence_step,
    router_probs_capture_keys,
)
from param_decomp.lm.batch import LMBatch, LMBatchWithRouting
from param_decomp.targets.lm_output import (
    MaterializedOutputEdge,
    OutputEdge,
    StreamedLinearOutput,
    StreamedOutputEdge,
)
from param_decomp.targets.qwen36_moe import (
    Qwen36MoeConfig,
    full_site_cs,
    qwen36_moe_site_specs,
)
from param_decomp.targets.testing import (
    tiny_qwen36_cfg,
    tiny_qwen36_decomposed_model,
    tiny_qwen36_moe_ci_fn_arch,
)

pytestmark = [
    pytest.mark.multidevice,
    pytest.mark.skipif(
        jax.default_backend() != "cpu" or jax.device_count() < 2,
        reason="requires a two-device CPU topology from make test-multidevice",
    ),
]

# Expert counts and dense component widths tile both tested TP sizes.
SITE_CS: dict[str, int] = {
    "experts_gate": 8,
    "experts_up": 8,
    "experts_down": 8,
    "shared_gate": 4,
    "shared_up": 4,
    "shared_down": 4,
}
BATCH, SEQ = 4, 8
OUTPUT_EDGES = pytest.mark.parametrize(
    "output_edge",
    [MaterializedOutputEdge(), StreamedOutputEdge(n_vocab_chunks=4)],
    ids=["materialized", "streamed"],
)


@dataclasses.dataclass(frozen=True)
class _PlacedSeat:
    cfg: Qwen36MoeConfig
    sites: Any
    mesh: Mesh
    placed: PlacedQwen
    components: Any
    ci_fn: CIFn[Any]
    ci_capture_keys: CaptureKeys
    tokens: jax.Array


def _placed_seat(output_edge: OutputEdge, tp: int) -> _PlacedSeat:
    """The tiny qwen36 target on a data=2 explicit mesh with authored TP."""
    cfg = tiny_qwen36_cfg()
    sites = qwen36_moe_site_specs(cfg, full_site_cs(cfg, SITE_CS))
    model = dataclasses.replace(
        tiny_qwen36_decomposed_model(cfg, sites, random.PRNGKey(0)), output_edge=output_edge
    )
    mesh = Mesh(
        np.asarray(jax.devices()[: 2 * tp]).reshape(2, tp),
        ("data", "tp"),
        axis_types=(AxisType.Explicit,) * 2,
    )
    rules = from_config("zero1-replicated-resident-moe", mesh, sites)
    ci_fn = tiny_qwen36_moe_ci_fn_arch(model).initialize(model.sites, rules, random.PRNGKey(2))
    with jax.set_mesh(mesh):
        return _PlacedSeat(
            cfg=cfg,
            sites=sites,
            mesh=mesh,
            placed=place_target(model, rules),
            components=init_component_stacks_placed(sites, random.PRNGKey(1), rules),
            ci_fn=jax.device_put(ci_fn, ci_fn.shardings(mesh)),
            ci_capture_keys=ci_fn.capture_keys,
            tokens=shard_batch(
                random.randint(random.PRNGKey(3), (BATCH, SEQ), 0, cfg.vocab_size),
                mesh,
                batch_axis=0,
            ),
        )


@OUTPUT_EDGES
@pytest.mark.parametrize("tp", [1, 2])
def test_batch_context_builds_and_scores_on_either_output_edge(output_edge: OutputEdge, tp: int):
    if jax.device_count() < 2 * tp:
        pytest.skip(f"requires {2 * tp} local devices")
    seat = _placed_seat(output_edge, tp)
    cfg, mesh, placed = seat.cfg, seat.mesh, seat.placed
    with jax.set_mesh(mesh):
        context_step = jax.jit(
            make_lm_batch_context_step(placed, seat.ci_capture_keys, EMPTY_CAPTURE_KEYS, mesh)
        )
        forward = context_step(placed, seat.components, seat.ci_fn, LMBatch(seat.tokens))
        clean_output = forward.clean_output
        ci = forward.ci
        conditioning = forward.conditioning
        assert isinstance(conditioning, LMBatchWithRouting)
        assert any(isinstance(value, SelectedCI) for value in ci.lower.values())
        assert conditioning.selection.indices.shape == (
            cfg.n_layer,
            BATCH,
            SEQ,
            cfg.n_experts_per_token,
        )
        match output_edge:
            case MaterializedOutputEdge():
                assert isinstance(clean_output, jax.Array)
                assert clean_output.shape == (BATCH, SEQ, cfg.vocab_size)
            case StreamedOutputEdge(n_vocab_chunks=n_vocab_chunks):
                assert isinstance(clean_output, StreamedLinearOutput)
                assert clean_output.activations.shape == (BATCH, SEQ, cfg.n_embd)
                assert clean_output.n_chunks == n_vocab_chunks

        context = LMBatchContext(
            pass_index=0,
            batch_index=0,
            forward=forward,
            persistent_sources={},
        )
        metrics = make_ce_kl_scorer(placed, 0.5, mesh)(
            placed, prepared_batch_from_context(context, EMPTY_CAPTURE_KEYS), random.PRNGKey(4)
        )
    assert metrics.keys() >= {"ce_kl/kl_ci_masked", "ce_kl/ce_difference_unmasked"}
    assert all(np.isfinite(np.asarray(value)).all() for value in metrics.values())


@OUTPUT_EDGES
def test_router_divergence_step_runs_every_strategy_placed(output_edge: OutputEdge):
    if jax.device_count() < 4:
        pytest.skip("requires four devices for data=2, tp=2")
    seat = _placed_seat(output_edge, tp=2)
    cfg, mesh, placed = seat.cfg, seat.mesh, seat.placed
    metric = RouterDivergenceConfig(
        strategies=(
            CIMaskedStrategy(),
            StochasticStrategy(),
            FreshPGDStrategy(n_steps=1, step_size=0.1),
            PersistentStrategy(state_key="ppgd"),
        )
    )
    with jax.set_mesh(mesh):
        probs_keys = router_probs_capture_keys(placed)
        context_step = jax.jit(
            make_lm_batch_context_step(placed, seat.ci_capture_keys, frozenset(probs_keys), mesh)
        )
        forward = context_step(placed, seat.components, seat.ci_fn, LMBatch(seat.tokens))
        clean_output = forward.clean_output
        captures = forward.captures
        ci = forward.ci
        prepared_weights = forward.prepared_weights
        conditioning = forward.conditioning
        persistent_sources = {
            "ppgd": init_sources_sharded(
                seat.sites,
                Positioned(n_positions=SEQ),
                "bc",
                2 * BATCH,
                jnp.dtype(jnp.float32),
                random.PRNGKey(7),
                mesh,
            )
        }
        results = jax.jit(
            make_router_divergence_step(
                placed,
                metric,
                mesh,
            )
        )(
            placed,
            prepared_weights,
            conditioning,
            ci.lower,
            {probs_key: captures[probs_key] for probs_key in probs_keys},
            clean_output,
            persistent_sources,
            random.PRNGKey(9),
        )
    assert results.keys() == {"ci_masked", "stochastic", "fresh_pgd", "persistent"}
    for kind, (sums, n_tokens) in results.items():
        assert n_tokens == BATCH * SEQ, kind
        for distance in ("kl", "topk_overlap", "weight_mae"):
            value = np.asarray(getattr(sums, distance))
            assert value.shape == (cfg.n_layer,) and np.isfinite(value).all(), (kind, distance)
        assert (np.asarray(sums.topk_overlap) <= n_tokens).all(), kind
