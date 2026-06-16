"""CPU tests for the torch-layout exporter (`export.py`).

These pin the pure mapping (key names, per-site V/U reads, the site-order permutation,
frozen-key rename) — including attention (q/k/v/o) sites with heterogeneous C. The
cross-framework numeric proof lived in the torch tools pair (`gen_export_fixture.py` +
`verify_export_torch.py`), now deleted so `param_decomp_jax` imports no torch; the
committed `tools/export_fixtures/*` are the frozen goldens (resurrect the torch
verifier from the `torch-oracle` git tag if the mapping changes).
"""

import numpy as np
from jax import random

from jax_single_pool.ci_fn import CIArch, init_ci_fn
from jax_single_pool.export import (
    ci_fn_state,
    components_state,
    frozen_target_keys,
    torch_site_order,
)
from jax_single_pool.llama8b import (
    MLP_KINDS,
    canonical_site_cs,
    init_decomp_vu,
    llama_site_specs,
    mlp_family_site_cs,
    site_name,
)
from jax_single_pool.lm import SiteC, SiteSpec
from jax_single_pool.tests.test_llama8b import _tiny_cfg

_C = 6
_ARCH = CIArch(d_model=16, n_blocks=2, n_heads=2, mlp_hidden=24)


def _mlp_sites(first: int, last: int) -> tuple[SiteSpec, ...]:
    return llama_site_specs(_tiny_cfg(), mlp_family_site_cs(first, last, _C))


_MIXED_SITE_CS = canonical_site_cs(
    (
        SiteC("layers.4.self_attn.q_proj", 4),
        SiteC("layers.4.self_attn.o_proj", 6),
        SiteC("layers.4.mlp.gate_proj", 8),
    )
)


def test_torch_site_order_is_lexicographic():
    names = tuple(site_name(layer, kind) for layer in (2, 10) for kind in MLP_KINDS)
    # torch sorts module paths as STRINGS: "layers.10..." < "layers.2...", and within a
    # layer (down, gate, up) — not the JAX (gate, up, down).
    assert torch_site_order(names) == (
        "layers.10.mlp.down_proj",
        "layers.10.mlp.gate_proj",
        "layers.10.mlp.up_proj",
        "layers.2.mlp.down_proj",
        "layers.2.mlp.gate_proj",
        "layers.2.mlp.up_proj",
    )
    # within a layer, "mlp" < "self_attn" and attention kinds sort (k, o, q, v)
    mixed = tuple(s.name for s in llama_site_specs(_tiny_cfg(), _MIXED_SITE_CS))
    assert torch_site_order(mixed) == (
        "layers.4.mlp.gate_proj",
        "layers.4.self_attn.o_proj",
        "layers.4.self_attn.q_proj",
    )


def test_components_state_reads_per_site():
    sites = _mlp_sites(4, 5)
    vu = init_decomp_vu(sites, random.PRNGKey(0))
    state = components_state(vu, sites)

    assert set(state) == {f"model.{s.name}.components.{p}" for s in sites for p in ("V", "U")}
    for spec in sites:
        V, U = vu.site(spec.name)
        np.testing.assert_array_equal(state[f"model.{spec.name}.components.V"], np.asarray(V))
        np.testing.assert_array_equal(state[f"model.{spec.name}.components.U"], np.asarray(U))
    # torch LinearComponents stores V (d_in, C), U (C, d_out).
    assert state["model.layers.4.mlp.gate_proj.components.V"].shape == (32, _C)
    assert state["model.layers.4.mlp.gate_proj.components.U"].shape == (_C, 64)
    assert state["model.layers.4.mlp.down_proj.components.V"].shape == (64, _C)
    assert state["model.layers.4.mlp.down_proj.components.U"].shape == (_C, 32)


def test_components_state_attention_sites_heterogeneous_c():
    cfg = _tiny_cfg()
    sites = llama_site_specs(cfg, _MIXED_SITE_CS)
    vu = init_decomp_vu(sites, random.PRNGKey(0))
    state = components_state(vu, sites)
    qd = cfg.n_head * cfg.head_dim
    assert state["model.layers.4.self_attn.q_proj.components.V"].shape == (cfg.n_embd, 4)
    assert state["model.layers.4.self_attn.q_proj.components.U"].shape == (4, qd)
    assert state["model.layers.4.self_attn.o_proj.components.V"].shape == (qd, 6)
    assert state["model.layers.4.self_attn.o_proj.components.U"].shape == (6, cfg.n_embd)
    assert state["model.layers.4.mlp.gate_proj.components.V"].shape == (cfg.n_embd, 8)


def _block_bounds(order: tuple[str, ...], sizes: dict[str, int], site: str) -> slice:
    offset = sum(sizes[s] for s in order[: order.index(site)])
    return slice(offset, offset + sizes[site])


def _assert_ci_permutation(sites: tuple[SiteSpec, ...]) -> None:
    ci_fn = init_ci_fn(_ARCH, sites, random.PRNGKey(1))
    state = ci_fn_state(ci_fn, sites)
    jax_order = tuple(s.name for s in sites)
    sorted_order = torch_site_order(jax_order)
    assert jax_order != sorted_order

    d_in = {s.name: s.d_in for s in sites}
    c = {s.name: s.C for s in sites}
    in_proj = np.asarray(ci_fn.in_proj_w)
    out_w = np.asarray(ci_fn.out_w)
    out_b = np.asarray(ci_fn.out_b)
    for site in jax_order:
        np.testing.assert_array_equal(
            state["ci_fn._global_ci_fn._input_projector.W"][
                _block_bounds(sorted_order, d_in, site)
            ],
            in_proj[_block_bounds(jax_order, d_in, site)],
        )
        np.testing.assert_array_equal(
            state["ci_fn._global_ci_fn._output_head.W"][:, _block_bounds(sorted_order, c, site)],
            out_w[:, _block_bounds(jax_order, c, site)],
        )
        np.testing.assert_array_equal(
            state["ci_fn._global_ci_fn._output_head.b"][_block_bounds(sorted_order, c, site)],
            out_b[_block_bounds(jax_order, c, site)],
        )


def test_ci_fn_state_keys_and_permutation():
    sites = _mlp_sites(4, 5)
    ci_fn = init_ci_fn(_ARCH, sites, random.PRNGKey(1))
    state = ci_fn_state(ci_fn, sites)

    prefix = "ci_fn._global_ci_fn"
    expected_keys = {f"{prefix}._input_projector.{p}" for p in ("W", "b")}
    expected_keys |= {f"{prefix}._output_head.{p}" for p in ("W", "b")}
    for i in range(_ARCH.n_blocks):
        expected_keys |= {
            f"{prefix}._blocks.{i}.attn.{name}_proj.weight" for name in ("q", "k", "v", "out")
        }
        expected_keys.add(f"{prefix}._blocks.{i}.attn.rope.inv_freq")
        expected_keys |= {f"{prefix}._blocks.{i}.mlp.{j}.{p}" for j in (0, 2) for p in ("W", "b")}
    assert set(state) == expected_keys

    _assert_ci_permutation(sites)


def test_ci_fn_permutation_with_attention_sites():
    """JAX computation order (q, o, gate) vs torch lexicographic (gate, o, q) — the
    permutation must hold for heterogeneous d_in AND heterogeneous C blocks."""
    _assert_ci_permutation(llama_site_specs(_tiny_cfg(), _MIXED_SITE_CS))


def test_ci_fn_state_block_weights_unpermuted_fp32():
    sites = _mlp_sites(4, 4)
    ci_fn = init_ci_fn(_ARCH, sites, random.PRNGKey(2))
    state = ci_fn_state(ci_fn, sites)
    for i, block in enumerate(ci_fn.blocks):
        prefix = f"ci_fn._global_ci_fn._blocks.{i}"
        np.testing.assert_array_equal(state[f"{prefix}.attn.q_proj.weight"], np.asarray(block.wq))
        np.testing.assert_array_equal(state[f"{prefix}.attn.out_proj.weight"], np.asarray(block.wo))
        np.testing.assert_array_equal(state[f"{prefix}.mlp.0.W"], np.asarray(block.w1))
        np.testing.assert_array_equal(state[f"{prefix}.mlp.2.b"], np.asarray(block.b2))
        np.testing.assert_array_equal(
            state[f"{prefix}.attn.rope.inv_freq"], np.asarray(ci_fn.inv_freq)
        )
    assert all(v.dtype == np.float32 for v in state.values())


def test_frozen_target_keys_rename():
    decomposed = frozenset(
        {site_name(18, "gate"), site_name(18, "up"), site_name(18, "down"), site_name(18, "q")}
    )
    keys = frozen_target_keys(n_layer=20, decomposed_sites=decomposed)
    # Decomposed sites: frozen weight renames to the ComponentLinear buffer — for
    # attention sites exactly like MLP ones.
    assert (
        keys["model.layers.18.mlp.gate_proj.target_weight"]
        == "model.layers.18.mlp.gate_proj.weight"
    )
    assert "model.layers.18.mlp.gate_proj.weight" not in keys
    assert (
        keys["model.layers.18.self_attn.q_proj.target_weight"]
        == "model.layers.18.self_attn.q_proj.weight"
    )
    assert "model.layers.18.self_attn.q_proj.weight" not in keys
    # Non-decomposed matrices keep `.weight`; lm_head gains the `model.` prefix.
    assert (
        keys["model.layers.18.self_attn.k_proj.weight"] == "model.layers.18.self_attn.k_proj.weight"
    )
    assert keys["model.layers.19.mlp.gate_proj.weight"] == "model.layers.19.mlp.gate_proj.weight"
    assert keys["model.lm_head.weight"] == "lm_head.weight"
    assert keys["model.embed_tokens.weight"] == "model.embed_tokens.weight"
    # 1 embed + 20 layers x (2 norms + 4 attn + 3 mlp) + final norm + lm_head.
    assert len(keys) == 1 + 20 * 9 + 2
