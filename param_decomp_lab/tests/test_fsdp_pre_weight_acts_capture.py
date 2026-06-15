"""Regression: multi-layer `pre_weight_acts` capture under FSDP2 `fully_shard`.

A wrapped block reconstructs its forward args (via `tree_unflatten`) whenever a tensor arg
requires grad. The old capture threaded a mutable output dict THROUGH the block forward, so
that reconstruction handed every block past the first decomposed one a fresh copy of the dict
and silently dropped its sites — the CI fn's `layer_order` then KeyError'd on the first missing
site (e.g. `layers.21.mlp.down_proj` on the 12-layer Llama bench). The fix stashes each leaf's
act on the module instead of threading it (`_ComponentModule` / Llama `ComponentLinear` +
`capture_acts`), which the arg reconstruction can't touch.

Reproduces only under real `fully_shard` on CUDA (the CPU device-mesh path doesn't trigger the
grad-requiring arg reconstruction), so the test is GPU-gated and runs a single-rank world.
"""

import os
from collections.abc import Iterator

import pytest
import torch
import torch.distributed as dist
from torch.distributed.fsdp import fully_shard

from param_decomp.components import make_components
from param_decomp_lab.experiments.lm.vendored.llama_3_1.components import (
    ComponentLlama,
    componentize_llama,
)
from param_decomp_lab.experiments.lm.vendored.llama_3_1.config import VendoredLlamaConfig
from param_decomp_lab.experiments.lm.vendored.llama_3_1.model import VendoredLlama

DECOMPOSED_LAYERS = (1, 2)
PROJS = ("gate_proj", "up_proj", "down_proj")
EXPECTED_SITES = {f"layers.{layer}.mlp.{proj}" for layer in DECOMPOSED_LAYERS for proj in PROJS}


def _tiny_componentized_llama() -> ComponentLlama:
    cfg = VendoredLlamaConfig(
        model_type="VendoredLlama",
        max_position_embeddings=128,
        vocab_size=64,
        n_layer=4,
        n_head=4,
        n_key_value_heads=2,
        n_embd=32,
        n_intermediate=64,
        rope_scaling=None,
        rms_norm_eps=1e-5,
    )
    torch.manual_seed(0)
    model = VendoredLlama(cfg)
    components = make_components(model, {site: 4 for site in EXPECTED_SITES})
    return componentize_llama(model, components)


@pytest.fixture
def single_rank_pg() -> Iterator[None]:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29577")
    dist.init_process_group("nccl", rank=0, world_size=1)
    torch.cuda.set_device(0)
    try:
        yield
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FSDP2 capture bug only repros on CUDA")
@pytest.mark.usefixtures("single_rank_pg")
def test_fully_shard_captures_all_multilayer_sites() -> None:
    model = _tiny_componentized_llama()
    for block in model._layers:
        fully_shard(block)
    model = model.to("cuda")
    idx = torch.randint(0, 64, (2, 8), device="cuda")

    _, pre_acts = model.forward_with_pre_weight_acts(idx)
    assert set(pre_acts) == EXPECTED_SITES, f"dropped {sorted(EXPECTED_SITES - set(pre_acts))}"

    with model.use_cached_residual(idx):
        _, resid_start_acts = model.forward_with_pre_weight_acts(idx)
    assert set(resid_start_acts) == EXPECTED_SITES, (
        f"residual-start dropped {sorted(EXPECTED_SITES - set(resid_start_acts))}"
    )

    _, out_acts = model.forward_with_output_acts(idx)
    assert set(out_acts) == EXPECTED_SITES, (
        f"output-acts dropped {sorted(EXPECTED_SITES - set(out_acts))}"
    )
