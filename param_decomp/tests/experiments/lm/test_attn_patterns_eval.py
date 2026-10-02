"""CPU tests for the in-loop attention-pattern recon eval metrics.

Pins the target-owned `attention_pattern_from_qk` recipe (shape + causal/softmax sanity on both LM
targets), the all-false-routes clean target (KL=0 when masked==clean), and the
host-side token-weighted accumulation (combined = Σ sum_kl / Σ n_distributions).
"""

from collections.abc import Callable, Mapping
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh

from param_decomp.core.ci_fn.implementations.chunkwise.arch import (
    Chunk,
    ChunkwiseTransformerCIFnArch,
)
from param_decomp.core.ci_fn.implementations.transformer.layers import MHACIFnAttention
from param_decomp.core.ci_fn.interface import CIFn
from param_decomp.core.components import (
    ComponentStacks,
    DenseFactorization,
    SiteC,
    SiteCI,
    SiteSpec,
    init_component_stacks,
)
from param_decomp.core.masking import materialize_masking
from param_decomp.core.model import (
    EMPTY_CAPTURE_KEYS,
    CaptureKeys,
    ForwardResult,
    Masking,
    MaterializedMasking,
    PlacedModel,
    SiteRoutes,
    StochasticMasking,
)
from param_decomp.core.placement import PlacementRules
from param_decomp.experiments.lm.attn_patterns_eval import (
    AttnPatternsStep,
    LayerKLReduction,
    attn_output_key_by_site,
    attn_patterns_log_entries,
    fold_layer_kl,
    make_ci_attn_patterns_step,
    make_stochastic_attn_patterns_step,
)
from param_decomp.experiments.lm.load_run import PlacedTransformer
from param_decomp.lm.batch import LMBatchWithDocuments
from param_decomp.sequence import SequenceLayout
from param_decomp.targets.llama_simple_mlp import (
    canonical_site_cs as simple_canonical,
)
from param_decomp.targets.llama_simple_mlp import (
    site_specs as simple_site_specs,
)
from param_decomp.targets.lm_output import LMOutput
from param_decomp.targets.testing import (
    tiny_glu_cfg as _llama_cfg,
)
from param_decomp.targets.testing import (
    tiny_glu_decomposed_lm as _llama_decomposed_lm,
)
from param_decomp.targets.testing import (
    tiny_simple_mlp_cfg as _simple_cfg,
)
from param_decomp.targets.testing import (
    tiny_simple_mlp_decomposed_model as _simple_decomposed_model,
)
from param_decomp.targets.transformer import (
    TransformerPreparedMasking,
    TransformerPreparedWeights,
    glu_site_specs,
)
from param_decomp.tests.sequence import unsegmented_sequence_layout


def _build_ci_fn(model: PlacedTransformer, n_embd: int, key: jax.Array) -> CIFn[Any]:
    """One transformer chunk over all sites, reading the residual entering the first
    decomposed block. The old `CIArch(16, 1, 2, 32)` dims map onto the chunk arch."""
    site_names = model.model.site_names
    first_block = min(int(name.split(".")[1]) for name in site_names)
    arch = ChunkwiseTransformerCIFnArch(
        chunks=(Chunk(input_taps=(f"resid.{first_block}",), output_sites=site_names),),
        input_dim=n_embd,
        d_model=16,
        n_blocks=1,
        attention=MHACIFnAttention(mask="bidirectional", implementation="xla", n_heads=2),
        ffn_hidden=32,
        ffn_kind="gelu",
        learned_norm_scale=False,
    )
    return arch.initialize(model.model.sites, None, key)


def test_attention_pattern_from_qk_shape_and_causal_softmax_llama():
    cfg = _llama_cfg()
    sites = glu_site_specs(cfg, (SiteC("layers.0.self_attn.q_proj", 4),))
    model = _llama_decomposed_lm(cfg, sites, jax.random.PRNGKey(0))
    b, t = 2, 9
    qd, kvd = cfg.n_head * cfg.head_dim, cfg.n_kv_head * cfg.head_dim
    q = jax.random.normal(jax.random.PRNGKey(1), (b, t, qd))
    k = jax.random.normal(jax.random.PRNGKey(2), (b, t, kvd))
    pattern = np.asarray(
        model.attention_pattern_from_qk(
            "layers.0.self_attn.q_proj", q, k, SequenceLayout(jnp.zeros((b, t), dtype=jnp.int32))
        )
    )

    assert pattern.shape == (b, cfg.n_head, t, t)
    np.testing.assert_allclose(pattern.sum(-1), 1.0, rtol=1e-5, atol=1e-5)
    upper = np.triu(np.ones((t, t), bool), k=1)
    assert np.allclose(pattern[:, :, upper], 0.0), "future positions must carry zero mass"
    assert pattern.dtype == np.float32


def test_attention_pattern_from_qk_shape_and_causal_softmax_simple_mlp():
    cfg = _simple_cfg()
    sites = simple_site_specs(cfg, (SiteC("h.0.attn.q_proj", 4),))
    model = _simple_decomposed_model(cfg, sites, jax.random.PRNGKey(0))
    b, t = 2, 7
    qd, kvd = cfg.n_head * cfg.head_dim, cfg.n_kv_head * cfg.head_dim
    q = jax.random.normal(jax.random.PRNGKey(1), (b, t, qd))
    k = jax.random.normal(jax.random.PRNGKey(2), (b, t, kvd))
    pattern = np.asarray(
        model.attention_pattern_from_qk(
            "h.0.attn.q_proj", q, k, SequenceLayout(jnp.zeros((b, t), dtype=jnp.int32))
        )
    )

    assert pattern.shape == (b, cfg.n_head, t, t)
    np.testing.assert_allclose(pattern.sum(-1), 1.0, rtol=1e-5, atol=1e-5)
    upper = np.triu(np.ones((t, t), bool), k=1)
    assert np.allclose(pattern[:, :, upper], 0.0)


def _context_values(
    model: PlacedTransformer,
    components: ComponentStacks,
    ci_fn: CIFn[Any],
    tokens: jax.Array,
) -> tuple[Any, Any, Mapping[str, SiteCI], dict[str, jax.Array]]:
    """The shared batch-context values the steps consume, prepared eagerly for tests."""
    output_key_by_site = attn_output_key_by_site(model)
    capture_keys = ci_fn.capture_keys | frozenset(output_key_by_site.values())
    clean = model.clean_forward(
        LMBatchWithDocuments.from_unsegmented_sequences(tokens), capture_keys
    )
    prepared_weights = model.prepare_compute_weights(components)
    ci_lower = ci_fn.prepare()(
        {key: clean.captures[key] for key in ci_fn.capture_keys},
        clean.conditioning,
        prepared_weights,
        sequence=unsegmented_sequence_layout(
            {key: clean.captures[key] for key in ci_fn.capture_keys}
        ),
        remat=False,
    ).lower
    outputs = {site: clean.captures[key] for site, key in output_key_by_site.items()}
    return prepared_weights, clean.conditioning, ci_lower, outputs


def _run_step(
    step: AttnPatternsStep[
        LMBatchWithDocuments,
        TransformerPreparedWeights,
        LMBatchWithDocuments,
        TransformerPreparedMasking,
    ],
    model: PlacedTransformer,
    components: ComponentStacks,
    ci_fn: CIFn[Any],
    tokens: jax.Array,
    key: jax.Array,
) -> tuple[dict[str, jax.Array], dict[str, int]]:
    prepared_weights, conditioning, ci_lower, outputs = _context_values(
        model, components, ci_fn, tokens
    )
    return step(
        model,
        prepared_weights,
        conditioning,
        ci_lower,
        outputs,
        key,
    )


def _accumulate(
    step: AttnPatternsStep[
        LMBatchWithDocuments,
        TransformerPreparedWeights,
        LMBatchWithDocuments,
        TransformerPreparedMasking,
    ],
    model: PlacedTransformer,
    components: ComponentStacks,
    ci_fn: CIFn[Any],
    token_batches: list[jax.Array],
    base_key: jax.Array,
) -> dict[str, LayerKLReduction]:
    reductions: dict[str, LayerKLReduction] = {}
    for batch_idx, tokens in enumerate(token_batches):
        batch_sum, batch_n = _run_step(
            step, model, components, ci_fn, tokens, jax.random.fold_in(base_key, batch_idx)
        )
        reductions = fold_layer_kl(reductions, batch_sum, batch_n)
    return reductions


def _llama_attn_setup():
    cfg = _llama_cfg()
    site_cs = (
        SiteC("layers.4.self_attn.q_proj", 8),
        SiteC("layers.4.self_attn.k_proj", 6),
        SiteC("layers.5.self_attn.q_proj", 8),
        SiteC("layers.5.self_attn.k_proj", 6),
    )
    from param_decomp.targets.transformer import canonical_site_cs

    sites = glu_site_specs(cfg, canonical_site_cs(site_cs))
    model = PlacedModel(
        model=_llama_decomposed_lm(cfg, sites, jax.random.PRNGKey(0)), placement=None
    )
    components = init_component_stacks(sites, jax.random.PRNGKey(1))
    ci_fn = _build_ci_fn(model, cfg.n_embd, jax.random.PRNGKey(2))
    return cfg, model, components, ci_fn


def test_ci_step_clean_equals_masked_when_ci_all_one_gives_finite_kl():
    cfg, model, components, ci_fn = _llama_attn_setup()
    step = jax.jit(make_ci_attn_patterns_step(model))
    b, t = 2, 12
    residual = jax.random.randint(jax.random.PRNGKey(4), (b, t), 0, cfg.vocab_size)

    sum_kl, n_dist = _run_step(step, model, components, ci_fn, residual, jax.random.PRNGKey(0))

    q_sites = [s for s in model.model.site_names if s.endswith("q_proj")]
    assert set(sum_kl) == set(q_sites) == set(n_dist)
    for q in q_sites:
        assert int(n_dist[q]) == b * cfg.n_head * t
        assert np.isfinite(float(sum_kl[q]))
        assert float(sum_kl[q]) >= 0.0, "KL is non-negative"


def test_accumulate_is_token_weighted_and_combines():
    cfg, model, components, ci_fn = _llama_attn_setup()
    step = jax.jit(make_ci_attn_patterns_step(model))
    res_a = jax.random.randint(jax.random.PRNGKey(4), (2, 10), 0, cfg.vocab_size)
    res_b = jax.random.randint(jax.random.PRNGKey(5), (2, 10), 0, cfg.vocab_size)

    one = _accumulate(step, model, components, ci_fn, [res_a], jax.random.PRNGKey(0))
    other = _accumulate(step, model, components, ci_fn, [res_b], jax.random.PRNGKey(0))
    two = _accumulate(step, model, components, ci_fn, [res_a, res_b], jax.random.PRNGKey(0))
    for site in one:
        assert two[site].n_distributions == one[site].n_distributions + other[site].n_distributions
        np.testing.assert_allclose(
            two[site].sum_kl, one[site].sum_kl + other[site].sum_kl, rtol=1e-4, atol=1e-4
        )

    entries = attn_patterns_log_entries("CIMaskedAttnPatternsReconLoss", two)
    combined = entries["CIMaskedAttnPatternsReconLoss"]
    total_sum = sum(r.sum_kl for r in two.values())
    total_n = sum(r.n_distributions for r in two.values())
    np.testing.assert_allclose(combined, total_sum / total_n, rtol=1e-6)
    for site, r in two.items():
        np.testing.assert_allclose(
            entries[f"CIMaskedAttnPatternsReconLoss/{site}"],
            r.sum_kl / r.n_distributions,
            rtol=1e-6,
        )


def test_stochastic_step_runs_and_counts_one_distribution_per_query():
    cfg, model, components, ci_fn = _llama_attn_setup()
    step = jax.jit(make_stochastic_attn_patterns_step(model))
    b, t = 2, 8
    residual = jax.random.randint(jax.random.PRNGKey(4), (b, t), 0, cfg.vocab_size)

    sum_kl, n_dist = _run_step(step, model, components, ci_fn, residual, jax.random.PRNGKey(0))
    for q in (s for s in model.model.site_names if s.endswith("q_proj")):
        assert int(n_dist[q]) == b * cfg.n_head * t
        assert np.isfinite(float(sum_kl[q])) and float(sum_kl[q]) >= 0.0


def test_simple_mlp_step_runs_end_to_end():
    cfg = _simple_cfg()
    site_cs = simple_canonical((SiteC("h.0.attn.q_proj", 6), SiteC("h.0.attn.k_proj", 6)))
    sites = simple_site_specs(cfg, site_cs)
    model = PlacedModel(
        model=_simple_decomposed_model(cfg, sites, jax.random.PRNGKey(0)), placement=None
    )
    components = init_component_stacks(sites, jax.random.PRNGKey(1))
    ci_fn = _build_ci_fn(model, cfg.n_embd, jax.random.PRNGKey(2))
    step = jax.jit(make_ci_attn_patterns_step(model))
    b, t = 2, 10
    residual = jax.random.randint(jax.random.PRNGKey(4), (b, t), 0, cfg.vocab_size)

    sum_kl, n_dist = _run_step(step, model, components, ci_fn, residual, jax.random.PRNGKey(0))
    assert set(sum_kl) == {"h.0.attn.q_proj"}
    assert int(n_dist["h.0.attn.q_proj"]) == b * cfg.n_head * t
    assert np.isfinite(float(sum_kl["h.0.attn.q_proj"]))


class _PositionlessStub(eqx.Module):
    """A positionless model whose methods are never called — only exercises the
    LM-only `has_position_axis` guards (which fire at step construction). It binds the
    LM output edge so the LM step factories admit it up to that guard."""

    sites: tuple[SiteSpec, ...] = eqx.field(static=True)
    has_position_axis: bool = eqx.field(static=True)

    @property
    def site_names(self) -> tuple[str, ...]:
        return tuple(s.name for s in self.sites)

    def shardings(self, placement: PlacementRules) -> "_PositionlessStub":
        del placement
        raise AssertionError("positionless stub fn must not be called")

    @staticmethod
    def recon_loss_fn(masked_output: LMOutput, clean_output: LMOutput) -> jax.Array:
        del masked_output, clean_output
        raise AssertionError("positionless stub fn must not be called")

    @staticmethod
    def pin_output_batch(output: LMOutput, mesh: Mesh | None) -> LMOutput:
        del output, mesh
        raise AssertionError("positionless stub fn must not be called")

    def site_output_keys(self, sites: tuple[str, ...]) -> tuple[str, ...]:
        del sites
        raise AssertionError("positionless stub fn must not be called")

    def clean_forward(
        self,
        resid: Any,
        capture_keys: CaptureKeys = EMPTY_CAPTURE_KEYS,
        *,
        placement: PlacementRules | None,
    ) -> ForwardResult[LMOutput, LMBatchWithDocuments]:
        del resid, capture_keys, placement
        raise AssertionError("positionless stub fn must not be called")

    def prepare_compute_weights(self, vu: Any, placement: object | None) -> Any:
        del placement
        return vu

    def component_activation_forward(
        self,
        prepared_weights: Any,
        inputs: Any,
        /,
        *,
        sites: tuple[str, ...],
        capture_keys: CaptureKeys,
        placement: PlacementRules | None,
    ) -> tuple[ForwardResult[LMOutput, LMBatchWithDocuments], dict[str, SiteCI]]:
        del prepared_weights, inputs, sites, capture_keys, placement
        raise NotImplementedError

    @staticmethod
    def prepare_masking(masking: Masking) -> MaterializedMasking:
        return materialize_masking(masking)

    @staticmethod
    def prepare_stochastic_masking(
        ci: Mapping[str, SiteCI],
    ) -> Callable[[jax.Array], MaterializedMasking]:
        def draw(draw_key: jax.Array) -> MaterializedMasking:
            return materialize_masking(StochasticMasking(ci=ci, draw_key=draw_key))

        return draw

    def masked_forward(
        self,
        prepared_weights: Any,
        inputs: Any,
        /,
        *,
        masking: MaterializedMasking,
        routes: SiteRoutes | None,
        placement: PlacementRules | None,
        capture_keys: CaptureKeys = EMPTY_CAPTURE_KEYS,
        remat: bool,
    ) -> ForwardResult[LMOutput, LMBatchWithDocuments]:
        del prepared_weights, inputs, masking, routes, placement, capture_keys, remat
        raise AssertionError("positionless stub fn must not be called")

    def target_weight_sq_norms(self) -> dict[str, jax.Array]:
        raise AssertionError("positionless stub fn must not be called")

    def weight_deltas(self, vu: Any) -> dict[str, jax.Array]:
        del vu
        raise AssertionError("positionless stub fn must not be called")


def test_attn_patterns_steps_reject_positionless_target():
    """Attention patterns are causal maps over a sequence axis; both step constructors
    must fail loud against a positionless target. The position-axis
    guard fires before site/pattern inspection, so a dummy pattern fn is fine."""
    model = PlacedModel(
        model=_PositionlessStub(
            sites=(
                SiteSpec("linear1", DenseFactorization(d_in=5, d_out=2, C=8), "linear1"),
                SiteSpec("linear2", DenseFactorization(d_in=2, d_out=5, C=6), "linear2"),
            ),
            has_position_axis=False,
        ),
        placement=None,
    )
    assert not model.model.has_position_axis
    with pytest.raises(AssertionError, match="LM-only"):
        jax.jit(make_ci_attn_patterns_step(model))
    with pytest.raises(AssertionError, match="LM-only"):
        jax.jit(make_stochastic_attn_patterns_step(model))
