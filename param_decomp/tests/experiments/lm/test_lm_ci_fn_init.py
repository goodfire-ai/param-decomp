"""The LM CI fn initializer: a conditioned CI fn's placed init calibrates its readouts to
one clean pass of the whole batch, through the decomposition's own fresh components, and
refuses a batch holding padding, too few tokens, or a partial chunk."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from param_decomp.core.ci_fn.implementations.transformer.backbone import BackboneCIFn
from param_decomp.core.ci_fn.implementations.transformer.component_conditioning import (
    CALIBRATION_CHUNK_TOKENS,
    ComponentConditioned,
)
from param_decomp.core.components import ComponentStacks
from param_decomp.core.placement import from_config
from param_decomp.core.precision import COMPUTE_DT
from param_decomp.core.run_state import init_decomposition
from param_decomp.core.sharding import hsdp_mesh, place_target, shard_batch
from param_decomp.experiments.lm.ci_fn_init import LMCIFnInitInputs, lm_ci_fn_initializer
from param_decomp.lm.batch import LMBatch, LMBatchWithDocuments
from param_decomp.sequence import SequenceLayout
from param_decomp.targets.testing import tiny_glu_cfg
from param_decomp.tests.core.test_ci_component_conditioning import (
    _batch,
    _conditioned_arch,
    _model,
    _reference_magnitude_quantile,
    _sites,
)

ONE_CHUNK_OF_ROWS = CALIBRATION_CHUNK_TOKENS // 16


def _placed_init(
    topology: tuple[int, int, int], batch: LMBatchWithDocuments
) -> tuple[ComponentConditioned, ComponentStacks]:
    """The placed conditioned init on `batch`, its conditioned backbone and components
    fetched to host."""
    sites = _sites()
    model = _model(sites)
    arch = _conditioned_arch(model)
    mesh = hsdp_mesh(*topology)
    rules = from_config("owner", mesh, sites)
    with jax.set_mesh(mesh):
        placed = place_target(model, rules)
        placed_batch = jax.tree.map(lambda x: shard_batch(x, mesh, batch_axis=0), batch)
        initializer = lm_ci_fn_initializer(
            arch, sites, rules, LMCIFnInitInputs(placed, placed_batch)
        )
        decomposition = init_decomposition(placed, initializer, jax.random.PRNGKey(0))
        declared = decomposition.ci_fn.shardings(mesh)
        for leaf, sharding in zip(
            jax.tree.leaves(decomposition.ci_fn), jax.tree.leaves(declared), strict=True
        ):
            assert leaf.sharding.is_equivalent_to(sharding, leaf.ndim), (leaf.sharding, sharding)
    fn = jax.device_get(decomposition.ci_fn)
    assert isinstance(fn, BackboneCIFn)
    assert isinstance(fn.backbone, ComponentConditioned)
    return fn.backbone, jax.device_get(decomposition.components)


def _assert_calibrated_at(topology: tuple[int, int, int]) -> None:
    """Calibration reads every row of the batch."""
    sites = _sites()
    model = _model(sites)
    arch = _conditioned_arch(model)
    batch = _batch(ONE_CHUNK_OF_ROWS)
    conditioned, components = _placed_init(topology, batch)

    clean = model.clean_forward(batch, arch.capture_keys, placement=None)
    for site in sites:
        readout = conditioned.readouts[site.name]
        x = clean.captures[readout.capture_key].astype(COMPUTE_DT)
        h = x @ components.site(site.name).V.astype(COMPUTE_DT)
        quantile = _reference_magnitude_quantile(h)
        for arm in (readout.positive, readout.negative):
            np.testing.assert_allclose(arm.input_scale, 1 / quantile, rtol=1e-2)


def test_conditioned_init_calibrates_to_the_clean_pass_on_one_device():
    _assert_calibrated_at((1, 1, 1))


@pytest.mark.multidevice
@pytest.mark.skipif(len(jax.devices()) != 8, reason="requires eight local devices")
def test_conditioned_init_calibrates_to_the_clean_pass_on_a_placed_mesh():
    _assert_calibrated_at((2, 2, 2))


def test_conditioned_init_refuses_a_calibration_batch_holding_padding():
    shape = (ONE_CHUNK_OF_ROWS, 16)
    tokens = jax.random.randint(jax.random.PRNGKey(4), shape, 0, tiny_glu_cfg().vocab_size)
    document_ids = jnp.zeros(shape, jnp.int32).at[1, 6:].set(-1)
    padded = LMBatchWithDocuments(LMBatch(tokens), SequenceLayout(document_ids))

    with pytest.raises(AssertionError, match="the calibration batch holds padding"):
        _placed_init((1, 1, 1), padded)


@pytest.mark.parametrize(
    ("n_rows", "message"),
    [(2, "fewer than calibration.min_n_tokens"), (5, "do not tile")],
)
def test_conditioned_init_refuses_a_calibration_batch_of_the_wrong_size(n_rows: int, message: str):
    """Two rows hold 32 tokens, under the 64 minimum; five hold 80, no whole number of
    calibration chunks."""
    with pytest.raises(AssertionError, match=message):
        _placed_init((1, 1, 1), _batch(n_rows))
