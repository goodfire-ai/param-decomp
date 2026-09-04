"""The LM global-MLP CI arch (`GlobalMlpCiConfig` -> `GlobalMLPCIArch`): tap resolution
over the whole site tree (one chunk spanning every decomposed block) and construction of
a positioned, pointwise-per-token CI fn on a tiny GLU target."""

import jax
import jax.numpy as jnp

from param_decomp.core.ci_fn import (
    ChunkwiseTransformerCIArch,
    GlobalMLPCIArch,
    GlobalMLPCIFn,
    TapSpec,
    build_ci_fn,
)
from param_decomp.experiments.lm.config import (
    ChunkwiseTransformerCiConfig,
    GlobalMlpCiConfig,
    GluTransformerCSpec,
    LayerList,
    SiteTree,
    resolve_lm_ci_arch,
    resolve_site_tree,
)
from param_decomp.targets import glu_transformer
from param_decomp.targets.glu_transformer import glu_site_specs
from param_decomp.targets.testing import tiny_glu_cfg
from param_decomp.targets.transformer_taps import TransformerTapGrammar
from param_decomp.vendored_jax.llama import LlamaConfig


def _grammar(cfg: LlamaConfig) -> TransformerTapGrammar:
    return TransformerTapGrammar(
        family=glu_transformer.FAMILY,
        n_layer=cfg.n_layer,
        d_resid=cfg.n_embd,
        d_attention_output=glu_transformer.site_dims(cfg, "o").d_in,
        d_mlp_hidden=glu_transformer.site_dims(cfg, "down").d_in,
        d_out_of=lambda name: (
            glu_transformer.site_dims(cfg, glu_transformer.FAMILY.parse(name)[1]).d_out
        ),
    )


def _tree(cfg: LlamaConfig) -> SiteTree:
    """Layers 2-3 at q/v/down: q and v SHARE their block's attention-input tap, the case
    the per-site-aligned arch shape could not represent."""
    spec = GluTransformerCSpec(layers=LayerList(indices=[2, 3]), cs={"q": 3, "v": 4, "down": 5})
    return resolve_site_tree(spec, glu_transformer.FAMILY, cfg.n_layer)


def test_resolves_all_block_taps_once_each_with_grammar_widths():
    cfg = tiny_glu_cfg()
    arch = resolve_lm_ci_arch(
        _tree(cfg),
        GlobalMlpCiConfig(hidden_dims=(16,), input_tap="all_block_taps"),
        _grammar(cfg),
    )
    assert isinstance(arch, GlobalMLPCIArch)
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
    grammar = _grammar(cfg)
    first = resolve_lm_ci_arch(
        _tree(cfg), GlobalMlpCiConfig(hidden_dims=(8,), input_tap="first_block_resid"), grammar
    )
    assert isinstance(first, GlobalMLPCIArch)
    assert first.input_taps == (TapSpec(key="resid.2", width=cfg.n_embd),)
    every = resolve_lm_ci_arch(
        _tree(cfg), GlobalMlpCiConfig(hidden_dims=(8,), input_tap="all_block_resids"), grammar
    )
    assert isinstance(every, GlobalMLPCIArch)
    assert every.input_taps == (
        TapSpec(key="resid.2", width=cfg.n_embd),
        TapSpec(key="resid.3", width=cfg.n_embd),
    )


def test_dispatcher_still_resolves_the_chunkwise_arm():
    cfg = tiny_glu_cfg()
    arch = resolve_lm_ci_arch(
        _tree(cfg),
        ChunkwiseTransformerCiConfig.model_validate(
            {
                "blocks_per_chunk": 1,
                "d_model": 16,
                "n_blocks": 1,
                "attention": {"kind": "mha", "n_heads": 2},
                "ffn": {"kind": "gelu", "hidden": 32},
            }
        ),
        _grammar(cfg),
    )
    assert isinstance(arch, ChunkwiseTransformerCIArch)


def test_built_ci_fn_gives_per_site_ci_per_position():
    cfg = tiny_glu_cfg()
    tree = _tree(cfg)
    arch = resolve_lm_ci_arch(
        tree, GlobalMlpCiConfig(hidden_dims=(16,), input_tap="all_block_taps"), _grammar(cfg)
    )
    assert isinstance(arch, GlobalMLPCIArch)
    sites = glu_site_specs(cfg, tree.site_cs(glu_transformer.FAMILY.name_of))
    ci_fn = build_ci_fn(arch, sites, jax.random.PRNGKey(0))
    assert isinstance(ci_fn, GlobalMLPCIFn)
    assert ci_fn.has_position_axis

    b, t = 2, 5
    taps = {
        tap.key: jax.random.normal(jax.random.fold_in(jax.random.PRNGKey(1), i), (b, t, tap.width))
        for i, tap in enumerate(arch.input_taps)
    }
    ci = ci_fn(taps, remat=False, placement=None)
    assert set(ci.lower) == {s.name for s in sites}
    for site in sites:
        assert ci.lower[site.name].shape == (b, t, site.C), site.name

    # Pointwise per token: perturbing one position moves no other position's CI.
    perturbed = dict(taps)
    first_key = arch.input_taps[0].key
    perturbed[first_key] = taps[first_key].at[:, 3, :].add(1.0)
    moved = ci_fn(perturbed, remat=False, placement=None)
    for site in sites:
        base, new = ci.preactivations[site.name], moved.preactivations[site.name]
        others = jnp.delete(new - base, 3, axis=1)
        assert not jnp.allclose(new[:, 3], base[:, 3]), site.name
        assert jnp.array_equal(others, jnp.zeros_like(others)), site.name
