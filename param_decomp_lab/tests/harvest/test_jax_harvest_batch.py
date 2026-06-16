"""The JAX->torch `HarvestBatch` bridge matches the torch `ParamDecompHarvestFn` shape
and semantics: lower-leaky CI as `causal_importance`, ‖U‖·(x@V) as
`component_activation`, firing = CI > threshold, int64 tokens."""

import jax.numpy as jnp
import numpy as np
import torch
from jax_single_pool.load_run import HarvestForward

from param_decomp_lab.harvest.scripts.run_worker_jax import harvest_batch_from_forward


def test_harvest_batch_from_forward_matches_torch_harvest_fn_semantics() -> None:
    rng = np.random.default_rng(0)
    tokens = rng.integers(0, 100, size=(2, 5)).astype(np.int32)
    ci = rng.uniform(0.0, 1.0, size=(2, 5, 3)).astype(np.float32)
    acts = rng.normal(size=(2, 5, 3)).astype(np.float32)
    probs = rng.uniform(size=(2, 5, 100)).astype(np.float32)
    fwd = HarvestForward(
        lower_leaky_ci={"h.0.mlp.c_fc": jnp.asarray(ci)},
        component_acts={"h.0.mlp.c_fc": jnp.asarray(acts)},
        output_probs=jnp.asarray(probs),
    )

    hb = harvest_batch_from_forward(tokens, fwd, activation_threshold=0.3)

    assert hb.tokens.dtype == torch.int64  # torch harvest path keys on long tokens
    assert torch.equal(hb.tokens, torch.from_numpy(tokens).long())
    site = "h.0.mlp.c_fc"
    assert torch.allclose(hb.activations[site]["causal_importance"], torch.from_numpy(ci))
    assert torch.allclose(hb.activations[site]["component_activation"], torch.from_numpy(acts))
    assert torch.equal(hb.firings[site], torch.from_numpy(ci) > 0.3)
    assert torch.allclose(hb.output_probs, torch.from_numpy(probs))
