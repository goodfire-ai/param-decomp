"""The LM global-MLP CI arch (`GlobalMlpCIFnConfig` -> `GlobalMLPCIFnArch`): tap resolution
over the whole site tree (one chunk spanning every decomposed block) and construction of
a positioned, pointwise-per-token CI fn on a tiny GLU target."""

import jax
import jax.numpy as jnp

from param_decomp.core.ci_fn.implementations.chunkwise.arch import ChunkwiseTransformerCIFnArch
from param_decomp.core.ci_fn.implementations.global_mlp import GlobalMLPCIFn, GlobalMLPCIFnArch
from param_decomp.core.ci_fn.interface import TapSpec
from param_decomp.core.components import init_component_stacks, require_full_emission
from param_decomp.experiments.lm.config import (
    ChunkwiseTransformerCIFnConfig,
    GlobalMlpCIFnConfig,
    GluTransformerCSpec,
    LayerList,
    ResolvedDecomposition,
    SiteTree,
    resolve_lm_ci_fn_arch,
    resolve_site_tree,
)
from param_decomp.experiments.lm.resolved import TargetConfig
from param_decomp.target_ports.llama import LlamaConfig
from param_decomp.targets import transformer
from param_decomp.targets.lm_output import MaterializedOutputEdge
from param_decomp.targets.testing import tiny_glu_cfg
from param_decomp.targets.transformer import glu_site_specs
from param_decomp.targets.transformer_taps import TransformerTapGrammar
from param_decomp.tests.sequence import unsegmented_sequence_layout


def _grammar(cfg: LlamaConfig) -> TransformerTapGrammar:
    return transformer.capture_grammar(
        transformer.GLU_ANATOMY,
        cfg.n_layer,
        cfg.n_embd,
        lambda kind: transformer.site_dims(cfg, kind),
    )


def _tree(cfg: LlamaConfig) -> SiteTree:
    """Layers 2-3 at q/v/down: q and v SHARE their block's attention-input tap, the case
    the per-site-aligned arch shape could not represent."""
    spec = GluTransformerCSpec(layers=LayerList(indices=[2, 3]), cs={"q": 3, "v": 4, "down": 5})
    return resolve_site_tree(spec, transformer.FAMILY, cfg.n_layer)


def _resolved(cfg: LlamaConfig) -> ResolvedDecomposition:
    tree = _tree(cfg)
    sites = tree.site_cs(transformer.FAMILY.name_of)
    target = TargetConfig(
        model_name="tiny",
        sites=sites,
        weights_dtype="float32",
        attention_implementation="xla",
        component_initialization="random",
        output_edge=MaterializedOutputEdge(),
    )
    return ResolvedDecomposition(target, tree, _grammar(cfg), glu_site_specs(cfg, sites))


def test_resolves_all_block_taps_once_each_with_grammar_widths():
    cfg = tiny_glu_cfg()
    arch = resolve_lm_ci_fn_arch(
        _resolved(cfg), GlobalMlpCIFnConfig(hidden_dims=(16,), input_tap="all_block_taps")
    )
    assert isinstance(arch, GlobalMLPCIFnArch)
    d, d_attn_out, d_hidden = cfg.n_embd, cfg.n_head * cfg.head_dim, cfg.n_intermediate
    assert arch.input_taps == tuple(
        TapSpec(key=f"{name}.{block}", width=width)
        for block in (2, 3)
        for name, width in (
            ("attn_in", d),
            ("attn_out", d_attn_out),
            ("mlp_in", d),
            ("mlp_hidden", d_hidden),
        )
    )
    assert arch.has_position_axis
    assert arch.hidden_dims == (16,)


def test_resolves_resid_taps_over_the_whole_tree():
    cfg = tiny_glu_cfg()
    resolved = _resolved(cfg)
    first = resolve_lm_ci_fn_arch(
        resolved, GlobalMlpCIFnConfig(hidden_dims=(8,), input_tap="first_block_resid")
    )
    assert isinstance(first, GlobalMLPCIFnArch)
    assert first.input_taps == (TapSpec(key="resid.2", width=cfg.n_embd),)
    every = resolve_lm_ci_fn_arch(
        resolved, GlobalMlpCIFnConfig(hidden_dims=(8,), input_tap="all_block_resids")
    )
    assert isinstance(every, GlobalMLPCIFnArch)
    assert every.input_taps == (
        TapSpec(key="resid.2", width=cfg.n_embd),
        TapSpec(key="resid.3", width=cfg.n_embd),
    )


def test_dispatcher_still_resolves_the_chunkwise_arm():
    cfg = tiny_glu_cfg()
    arch = resolve_lm_ci_fn_arch(
        _resolved(cfg),
        ChunkwiseTransformerCIFnConfig.model_validate(
            {
                "blocks_per_chunk": 1,
                "d_model": 16,
                "n_blocks": 1,
                "attention": {
                    "mask": "bidirectional",
                    "kind": "mha",
                    "implementation": "xla",
                    "n_heads": 2,
                },
                "ffn": {"kind": "gelu", "hidden": 32},
            }
        ),
    )
    assert isinstance(arch, ChunkwiseTransformerCIFnArch)


def test_built_ci_fn_gives_per_site_ci_per_position():
    cfg = tiny_glu_cfg()
    resolved = _resolved(cfg)
    arch = resolve_lm_ci_fn_arch(
        resolved, GlobalMlpCIFnConfig(hidden_dims=(16,), input_tap="all_block_taps")
    )
    assert isinstance(arch, GlobalMLPCIFnArch)
    sites = resolved.site_specs
    ci_fn = arch.initialize(sites, None, jax.random.PRNGKey(0))
    assert isinstance(ci_fn, GlobalMLPCIFn)
    assert ci_fn.has_position_axis
    components = init_component_stacks(sites, jax.random.PRNGKey(2))

    b, t = 2, 5
    taps = {
        tap.key: jax.random.normal(jax.random.fold_in(jax.random.PRNGKey(1), i), (b, t, tap.width))
        for i, tap in enumerate(arch.input_taps)
    }
    ci = ci_fn(taps, None, components, sequence=unsegmented_sequence_layout(taps), remat=False)
    assert set(ci.lower) == {s.name for s in sites}
    for site in sites:
        assert require_full_emission(ci.lower[site.name]).shape == (b, t, site.C), site.name

    # Pointwise per token: perturbing one position moves no other position's CI.
    perturbed = dict(taps)
    first_key = arch.input_taps[0].key
    perturbed[first_key] = taps[first_key].at[:, 3, :].add(1.0)
    moved = ci_fn(
        perturbed,
        None,
        components,
        sequence=unsegmented_sequence_layout(perturbed),
        remat=False,
    )
    for site in sites:
        base = require_full_emission(ci.preactivations[site.name])
        new = require_full_emission(moved.preactivations[site.name])
        others = jnp.delete(new - base, 3, axis=1)
        assert not jnp.allclose(new[:, 3], base[:, 3]), site.name
        assert jnp.array_equal(others, jnp.zeros_like(others)), site.name
