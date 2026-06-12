"""Torch side of the export round-trip verification (run in the param-decomp venv):

    /mnt/home/oli/param-decomp/.venv/bin/python jax_single_pool/tools/verify_export_torch.py

For each fixture case (`gen_export_fixture.py`: single-layer + two-layer), rebuilds the
REAL torch modules from the exported safetensors and checks against the recorded JAX
outputs:

  * key names — `GlobalSharedTransformerCiFn.load_state_dict(strict=True)`, plus the
    exported keys (incl. the fixture's frozen-target key list) must exactly equal the
    trainable+frozen split of a real tiny `LMComponentModel.state_dict()`.
  * per-site component forward `((x@V)*m)@U` through `LinearComponents.forward`.
  * the full CI fn forward (lower/upper leaky-hard), proving the in-proj/out-head
    site-order permutation.

Two known NUMERIC divergences (neither is a mapping error) sit between the JAX CI fn
and the production torch module; the strict rtol=2e-4 check runs against a
"numerics-matched" copy of the torch module that adopts the JAX choices, isolating the
key mapping / permutation under test, while the production-module deviation is measured
and reported:

  * GELU flavor — `jax.nn.gelu` defaults to the TANH approximation; torch's
    `TransformerBlock` uses exact-erf `nn.GELU()` (max pointwise gap ~4.7e-4).
  * weightless RMS-norm eps — JAX `CIFn` uses 1e-5; torch's weightless `F.rms_norm`
    defaults to `finfo(fp32).eps` ~ 1.19e-7.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch import nn

from param_decomp.ci_fns import (
    AttnConfig,
    GlobalCiConfig,
    GlobalSharedTransformerCiConfig,
    GlobalSharedTransformerCiFn,
    TargetLayerConfig,
)
from param_decomp.ci_sigmoids import SIGMOID_TYPES
from param_decomp.components import LinearComponents
from param_decomp.decomposition_targets import DecompositionTarget
from param_decomp_lab.experiments.lm.vendored.component_model import LMComponentModel
from param_decomp_lab.experiments.lm.vendored.llama_3_1.config import (
    Llama3RopeScaling,
    VendoredLlamaConfig,
)
from param_decomp_lab.experiments.lm.vendored.llama_3_1.model import VendoredLlama
from param_decomp_lab.three_pool.checkpoint import is_trainable_component_key

FIXTURE_DIR = Path(__file__).resolve().parent / "export_fixtures"
CASES = ("l18", "l20_21", "l18_attn")
RTOL, ATOL = 2e-4, 1e-6
CI_FN_PREFIX = "ci_fn._global_ci_fn."


def check(
    name: str, torch_out: torch.Tensor, ref: np.ndarray, rtol: float = RTOL, enforce: bool = True
) -> float:
    a = torch_out.detach().cpu().numpy().astype(np.float64)
    b = ref.astype(np.float64)
    abs_err = np.abs(a - b)
    worst = float((abs_err / (ATOL + rtol * np.abs(b))).max())
    max_rel = float((abs_err / np.maximum(np.abs(b), 1e-3)).max())
    print(f"    {name}: max_rel={max_rel:.3e}  worst_err/tol={worst:.3f}")
    if enforce:
        assert worst <= 1.0, f"{name}: exceeds rtol={rtol}/atol={ATOL} ({worst:.2f}x over)"
    return max_rel


def verify_key_parity(case: str, fixture: dict[str, np.ndarray], exported: set[str]) -> None:
    """Exported keys + the frozen-target key list must EXACTLY partition the state dict
    of a real tiny `LMComponentModel` with matching shape config."""
    _b, _t, n_layer, n_embd, n_intermediate, vocab = (int(v) for v in fixture["_dims"])
    d_model, n_blocks, n_heads, mlp_hidden = (int(v) for v in fixture["_arch"])
    site_names = [str(s) for s in fixture["_site_names"]]
    cs = [int(c) for c in fixture["_C"]]

    target = VendoredLlama(
        VendoredLlamaConfig(
            model_type="VendoredLlama",
            max_position_embeddings=512,
            vocab_size=vocab,
            n_layer=n_layer,
            n_head=2,
            n_key_value_heads=1,
            n_embd=n_embd,
            n_intermediate=n_intermediate,
            rope_theta=500000.0,
            rope_scaling=Llama3RopeScaling(),
            rms_norm_eps=1e-5,
        )
    )
    target.eval()
    target.requires_grad_(False)
    component_model = LMComponentModel.build(
        target_model=target,
        decomposition_targets=[
            DecompositionTarget(module_path=name, C=c)
            for name, c in zip(site_names, cs, strict=True)
        ],
        ci_config=GlobalCiConfig(
            fn_type="global_shared_transformer",
            simple_transformer_ci_cfg=GlobalSharedTransformerCiConfig(
                d_model=d_model,
                n_blocks=n_blocks,
                mlp_hidden_dim=[mlp_hidden],
                attn_config=AttnConfig(n_heads=n_heads, max_len=2048, rope_base=10000.0),
            ),
        ),
        sigmoid_type="leaky_hard",
    )
    expected = set(component_model.state_dict().keys())
    expected_trainable = {k for k in expected if is_trainable_component_key(k)}
    expected_frozen = expected - expected_trainable

    assert exported == expected_trainable, (
        f"{case}: exported keys != LMComponentModel trainable (V/U + ci_fn) keys\n"
        f"  missing: {sorted(expected_trainable - exported)}\n"
        f"  extra:   {sorted(exported - expected_trainable)}"
    )
    frozen_keys = {str(k) for k in fixture["_frozen_keys"]}
    assert frozen_keys == expected_frozen, (
        f"{case}: frozen-target keys != LMComponentModel frozen keys\n"
        f"  missing: {sorted(expected_frozen - frozen_keys)}\n"
        f"  extra:   {sorted(frozen_keys - expected_frozen)}"
    )
    print(f"    key parity: {len(exported | frozen_keys)} keys == full LMComponentModel state dict")


def verify_components(fixture: dict[str, np.ndarray], tensors: dict[str, torch.Tensor]) -> float:
    worst = 0.0
    for name, d_in, d_out, c in zip(
        fixture["_site_names"], fixture["_d_in"], fixture["_d_out"], fixture["_C"], strict=True
    ):
        site = str(name)
        components = LinearComponents(C=int(c), d_in=int(d_in), d_out=int(d_out), bias=None)
        with torch.no_grad():
            components.V.copy_(tensors[f"model.{site}.components.V"])
            components.U.copy_(tensors[f"model.{site}.components.U"])
            out = components(
                torch.tensor(fixture[f"x::{site}"]), mask=torch.tensor(fixture[f"mask::{site}"])
            )
        worst = max(worst, check(f"component fwd {site}", out, fixture[f"component_out::{site}"]))
    return worst


def build_ci_fn(
    fixture: dict[str, np.ndarray], tensors: dict[str, torch.Tensor]
) -> GlobalSharedTransformerCiFn:
    d_model, n_blocks, n_heads, mlp_hidden = (int(v) for v in fixture["_arch"])
    layer_configs = {
        str(name): TargetLayerConfig(input_dim=int(d_in), C=int(c))
        for name, d_in, c in zip(
            fixture["_site_names"], fixture["_d_in"], fixture["_C"], strict=True
        )
    }
    ci_fn = GlobalSharedTransformerCiFn(
        target_model_layer_configs=layer_configs,
        d_model=d_model,
        n_layers=n_blocks,
        n_heads=n_heads,
        max_len=2048,
        mlp_hidden_dims=[mlp_hidden],
        rope_base=10000.0,
    )
    ci_fn.load_state_dict(
        {k.removeprefix(CI_FN_PREFIX): v for k, v in tensors.items() if k.startswith(CI_FN_PREFIX)},
        strict=True,
    )
    return ci_fn.eval()


@contextmanager
def jax_numerics(ci_fn: GlobalSharedTransformerCiFn) -> Iterator[None]:
    """Adopt the JAX CI fn's numeric choices on the torch module: tanh GELU and
    rms-norm eps=1e-5 (see module docstring). Param-free, so the loaded state dict —
    the thing under test — is untouched."""
    original_gelus = [block.mlp[1] for block in ci_fn._blocks]
    assert all(isinstance(g, nn.GELU) for g in original_gelus)
    original_rms_norm = F.rms_norm

    def rms_norm_jax_eps(x, normalized_shape, weight=None, eps=None):
        return original_rms_norm(x, normalized_shape, weight, 1e-5 if eps is None else eps)

    try:
        for block in ci_fn._blocks:
            block.mlp[1] = nn.GELU(approximate="tanh")
        F.rms_norm = rms_norm_jax_eps
        yield
    finally:
        F.rms_norm = original_rms_norm
        for block, gelu in zip(ci_fn._blocks, original_gelus, strict=True):
            block.mlp[1] = gelu


def verify_ci_fn(
    fixture: dict[str, np.ndarray],
    ci_fn: GlobalSharedTransformerCiFn,
    label: str,
    rtol: float,
    enforce: bool = True,
) -> float:
    inputs = {str(name): torch.tensor(fixture[f"x::{name}"]) for name in fixture["_site_names"]}
    with torch.no_grad():
        logits = ci_fn(inputs)
        lower = torch.split(SIGMOID_TYPES["lower_leaky_hard"](logits), ci_fn.split_sizes, dim=-1)
        upper = torch.split(SIGMOID_TYPES["upper_leaky_hard"](logits), ci_fn.split_sizes, dim=-1)
    worst = 0.0
    for i, site in enumerate(ci_fn.layer_order):
        worst = max(
            worst,
            check(
                f"ci lower {site} [{label}]", lower[i], fixture[f"ci_lower::{site}"], rtol, enforce
            ),
        )
        worst = max(
            worst,
            check(
                f"ci upper {site} [{label}]", upper[i], fixture[f"ci_upper::{site}"], rtol, enforce
            ),
        )
    return worst


def main() -> None:
    for case in CASES:
        print(f"case {case}:")
        fixture = dict(np.load(FIXTURE_DIR / f"{case}.npz"))
        tensors = load_file(FIXTURE_DIR / f"{case}.safetensors")

        verify_key_parity(case, fixture, set(tensors))
        worst_component = verify_components(fixture, tensors)

        ci_fn = build_ci_fn(fixture, tensors)
        # Production module (exact GELU, default rms eps): measured, NEVER asserted —
        # these are the documented cross-framework numeric divergences, amplified on
        # tiny fixtures whose leaky-hard outputs sit near the clamp boundaries.
        production_rel = verify_ci_fn(
            fixture, ci_fn, "production numerics", rtol=5e-2, enforce=False
        )
        with jax_numerics(ci_fn):
            worst_ci = verify_ci_fn(fixture, ci_fn, "jax-matched numerics", rtol=RTOL)

        print(
            f"  PASS {case}: component max_rel={worst_component:.3e}, "
            f"ci-fn (jax-matched numerics) max_rel={worst_ci:.3e}, "
            f"ci-fn production-numerics divergence max_rel={production_rel:.3e}"
        )
    print("export round-trip verification PASSED")


if __name__ == "__main__":
    main()
