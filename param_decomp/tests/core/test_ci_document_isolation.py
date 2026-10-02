"""Document isolation under causal and bidirectional CI attention."""

from dataclasses import dataclass, replace
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from param_decomp.core.ci_fn.implementations.chunkwise.arch import ChunkwiseTransformerCIFnArch
from param_decomp.core.ci_fn.implementations.transformer.layers import (
    CIFnAttention,
    CIFnAttentionMask,
    GQACIFnAttention,
    MHACIFnAttention,
)
from param_decomp.core.ci_fn.interface import CIFn
from param_decomp.core.components import (
    ComponentStacks,
    SiteC,
    SiteSpec,
    init_component_stacks,
    site_ci_values,
)
from param_decomp.core.placement import from_config
from param_decomp.sequence import SequenceLayout
from param_decomp.targets.testing import (
    tiny_glu_cfg,
    tiny_glu_chunkwise_ci_fn_arch,
    tiny_glu_decomposed_lm,
)
from param_decomp.targets.transformer import KIND_ORDER, glu_site_specs, site_name
from param_decomp.tests.placed_ci_fn import placed_ci_fn


def _packed_layout() -> SequenceLayout:
    return SequenceLayout(
        jnp.asarray([[0, 0, 0, 0, 1, 1, 1, 1], [5, 5, 5, 5, 6, 6, 6, 6]], dtype=jnp.int32)
    )


@dataclass(frozen=True)
class _Case:
    ci_fn: CIFn[Any]
    components: ComponentStacks
    taps: dict[str, jax.Array]
    sequence: SequenceLayout
    site: str

    def values(self, taps: dict[str, jax.Array], sequence: SequenceLayout) -> jax.Array:
        result = self.ci_fn.prepare()(taps, None, self.components, sequence=sequence, remat=True)
        return site_ci_values(result.preactivations[self.site]).astype(jnp.float32)

    def slice(self, start: int, stop: int) -> "_Case":
        return replace(
            self,
            taps={name: values[:, start:stop] for name, values in self.taps.items()},
            sequence=SequenceLayout(self.sequence.document_ids[:, start:stop]),
        )


def _dense_arch(
    attention: CIFnAttention,
) -> tuple[ChunkwiseTransformerCIFnArch, tuple[SiteSpec, ...]]:
    cfg = replace(tiny_glu_cfg(), n_layer=2)
    sites = glu_site_specs(cfg, tuple(SiteC(site_name(0, kind), 4) for kind in KIND_ORDER))
    model = tiny_glu_decomposed_lm(cfg, sites, jax.random.PRNGKey(0))
    return replace(tiny_glu_chunkwise_ci_fn_arch(model, n_blocks=2), attention=attention), sites


def _dense_case(attention: CIFnAttention) -> _Case:
    arch, sites = _dense_arch(attention)
    fn = arch.initialize(sites, None, jax.random.PRNGKey(1))
    taps = {
        key: jax.random.normal(jax.random.PRNGKey(2), (2, 8, arch.input_dim))
        for key in arch.capture_keys
    }
    components = init_component_stacks(sites, jax.random.PRNGKey(3))
    return _Case(fn, components, taps, _packed_layout(), sites[0].name)


def _case(kind: Literal["mha", "gqa"], mask: CIFnAttentionMask) -> _Case:
    match kind:
        case "mha":
            return _dense_case(MHACIFnAttention(implementation="xla", n_heads=4, mask=mask))
        case "gqa":
            return _dense_case(
                GQACIFnAttention(implementation="xla", n_heads=4, n_kv_heads=2, mask=mask)
            )


@pytest.fixture(scope="module", params=["bidirectional", "causal"])
def mask(request: pytest.FixtureRequest) -> CIFnAttentionMask:
    return request.param


@pytest.fixture(scope="module", params=["mha", "gqa"])
def case(request: pytest.FixtureRequest, mask: CIFnAttentionMask) -> _Case:
    return _case(request.param, mask)


def test_packed_ci_matches_separate_documents(case: _Case) -> None:
    packed = case.values(case.taps, case.sequence)
    separate = []
    for start in (0, 4):
        document = case.slice(start, start + 4)
        separate.append(document.values(document.taps, document.sequence))
    np.testing.assert_allclose(packed, np.concatenate(separate, axis=1), rtol=1e-5, atol=1e-5)


def test_ci_future_dependencies_match_attention_mask(case: _Case, mask: CIFnAttentionMask) -> None:
    gradients = jax.grad(lambda taps: jnp.sum(case.values(taps, case.sequence)[:, 0]))(case.taps)
    for gradient in gradients.values():
        np.testing.assert_array_equal(gradient[:, 4:], 0)
    future_gradient = sum(
        float(jnp.linalg.norm(gradient[:, 1:4])) for gradient in gradients.values()
    )
    perturbed = {name: values.at[:, 1:4].set(-values[:, 1:4]) for name, values in case.taps.items()}
    before = case.values(case.taps, case.sequence)[:, 0]
    after = case.values(perturbed, case.sequence)[:, 0]
    match mask:
        case "bidirectional":
            assert future_gradient > 0
            assert not np.allclose(before, after)
        case "causal":
            assert future_gradient == 0
            np.testing.assert_array_equal(before, after)


def test_right_padding_cannot_change_valid_ci(case: _Case) -> None:
    document = case.slice(0, 4)
    expected = document.values(document.taps, document.sequence)
    sequence = SequenceLayout(case.sequence.document_ids.at[:, 4:].set(-1))
    perturbed = {
        name: values.at[:, 4:].set(jax.random.normal(jax.random.PRNGKey(19), values[:, 4:].shape))
        for name, values in case.taps.items()
    }
    np.testing.assert_allclose(
        case.values(perturbed, sequence)[:, :4], expected, rtol=1e-5, atol=1e-5
    )


def test_ci_rejects_mismatched_layout_shape(case: _Case) -> None:
    with pytest.raises(AssertionError):
        case.values(case.taps, SequenceLayout(jnp.zeros((1, 8), dtype=jnp.int32)))


def test_sequence_layout_resets_positions_and_masks_shifted_labels() -> None:
    sequence = SequenceLayout(
        jnp.asarray([[5, 5, 6, 6, 6, -1, -1], [2, 3, 3, 4, 4, 4, 4]], dtype=jnp.int32)
    )
    np.testing.assert_array_equal(
        sequence.position_resets(),
        [
            [True, False, True, False, False, True, False],
            [True, True, False, True, False, False, False],
        ],
    )
    np.testing.assert_array_equal(
        sequence.position_ids(), [[0, 1, 0, 1, 2, 0, 1], [0, 0, 1, 0, 1, 2, 3]]
    )
    np.testing.assert_array_equal(
        sequence.next_token_mask(),
        [[True, False, True, True, False, False], [False, True, False, True, True, True]],
    )
    mask = np.asarray(sequence.attention_mask())
    assert mask.shape == (2, 7, 7)
    assert mask[0, 0, 1] and mask[0, 1, 0]
    assert not mask[1, 0, 1]
    assert not mask[0, :5, 5:].any()
    assert not mask[0, 5:, :5].any()


@pytest.mark.multidevice
@pytest.mark.skipif(len(jax.devices()) < 4, reason="requires four local devices")
def test_gqa_document_isolation_with_batch_and_head_sharding(mask: CIFnAttentionMask) -> None:
    mesh = Mesh(
        np.asarray(jax.devices()[:4]).reshape(2, 2),
        ("data", "tp"),
        axis_types=(AxisType.Explicit,) * 2,
    )
    arch, sites = _dense_arch(
        GQACIFnAttention(implementation="xla", n_heads=4, n_kv_heads=2, mask=mask)
    )
    rules = from_config("zero1-replicated-resident", mesh, sites)
    sequence = SequenceLayout(
        jnp.asarray([[0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 1, 1, 1, 1, 1, 1]], dtype=jnp.int32)
    )
    taps = {
        key: jax.random.normal(jax.random.PRNGKey(2), (2, 8, arch.input_dim))
        for key in arch.capture_keys
    }
    unplaced = _Case(
        arch.initialize(sites, None, jax.random.PRNGKey(1)),
        init_component_stacks(sites, jax.random.PRNGKey(3)),
        taps,
        sequence,
        sites[0].name,
    )
    expected = unplaced.values(taps, sequence)
    batch_sharding = NamedSharding(mesh, P("data", None))
    with jax.set_mesh(mesh):
        fn = placed_ci_fn(arch, sites, jax.random.PRNGKey(1), mesh, rules)
        placed = replace(unplaced, ci_fn=fn)
        placed_taps = jax.device_put(taps, batch_sharding)
        placed_sequence = jax.device_put(sequence, batch_sharding)
        got = eqx.filter_jit(placed.values)(placed_taps, placed_sequence)
        gradients = eqx.filter_jit(
            jax.grad(lambda x: jnp.sum(placed.values(x, placed_sequence)[:, 0]))
        )(placed_taps)
    np.testing.assert_allclose(got, expected, rtol=2e-2, atol=2e-2)
    for gradient in gradients.values():
        np.testing.assert_array_equal(np.asarray(gradient)[0, 4:], 0)
        np.testing.assert_array_equal(np.asarray(gradient)[1, 2:], 0)
        if mask == "causal":
            np.testing.assert_array_equal(np.asarray(gradient)[:, 1:], 0)
