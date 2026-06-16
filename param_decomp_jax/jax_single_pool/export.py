"""Export an orbax `TrainState` checkpoint to the torch `LMComponentModel` layout.

    jsp-export <run_dir> [--step N]

Writes `<run_dir>/export/model_<step>.safetensors` carrying the vendored torch
`LMComponentModel`'s EXACT state-dict keys (`param_decomp_lab/experiments/lm/vendored/
component_model.py` @ `feature/fsdp-lm-trainer`), so the torch eval/harvest/postprocess
stack loads a JAX run like any 2/3-pool checkpoint. CPU-only; fp32 masters export as
fp32 (the frozen bf16 target upcasts exactly, matching torch's fp32 HF load).
Adversary sources and optimizer state are training-only and not exported.

Key mapping (was verified against the real torch modules by the now-deleted
`tools/verify_export_torch.py`; `tools/export_fixtures/*` are the frozen goldens):

  * V/U — read per site from `DecompVU.vu` to `model.<site>.components.{V,U}`
    (MLP and attention sites alike — torch componentizes any per-layer matrix).
    Same orientation both sides: V `(d_in, C)`, U `(C, d_out)` (torch
    `Components.__init__`). The frozen per-site weight goes to
    `model.<site>.target_weight` instead of `.weight`.
  * CI fn — `CIFn` fields map onto `ci_fn._global_ci_fn.*`:
    `in_proj_{w,b} -> _input_projector.{W,b}` and `out_{w,b} -> _output_head.{W,b}`
    (both stored `(d_in, d_out)`, applied `x @ W + b`, identical to JAX — no transpose);
    `blocks[i].w{q,k,v,o} -> _blocks.{i}.attn.{q,k,v,out}_proj.weight` (both stored
    `(d_out, d_in)` nn.Linear-style, applied `x @ W.T` — no transpose);
    `blocks[i].{w1,b1,w2,b2} -> _blocks.{i}.mlp.{0,2}.{W,b}` (`x @ W + b` both sides);
    `inv_freq -> _blocks.{i}.attn.rope.inv_freq` (persistent torch buffer, same formula).
  * SITE ORDER — the one real trap. Torch concatenates CI inputs/outputs in
    sorted-module-path order (per layer: mlp.{down,gate,up} then self_attn.{k,o,q,v});
    JAX uses computation order (`KIND_ORDER`: q,k,v,o,gate,up,down).
    `_concat_permutation` reorders the `in_proj` ROW blocks (by `d_in`) and the
    out-head COLUMN blocks (+ bias, by `C`) from the JAX order to `sorted(site_names)`.
"""

import argparse
from pathlib import Path

import jax
import numpy as np
from jax import random
from safetensors.numpy import save_file

from jax_single_pool.checkpoint import make_checkpoint_manager, restore_step
from jax_single_pool.ci_fn import CIFn
from jax_single_pool.config import TargetConfig
from jax_single_pool.llama8b import (
    KIND_ORDER,
    DecompVU,
    _hf_snapshot_dir,
    _HFWeights,
    llama31_8b_config,
    llama_decomposed_lm,
    llama_site_specs,
    site_name,
)
from jax_single_pool.lm import SiteSpec
from jax_single_pool.run_state import build_optimizers, init_train_state
from jax_single_pool.sharding import dp_mesh
from jax_single_pool.torch_config import load_run_dir_config

CI_FN_PREFIX = "ci_fn._global_ci_fn"


def _f32(a: object) -> np.ndarray:
    return np.asarray(a).astype(np.float32)


def torch_site_order(site_names: tuple[str, ...]) -> tuple[str, ...]:
    """Torch's canonical site order: `sorted()` module paths (LEXICOGRAPHIC — exactly
    `GlobalSharedTransformerCiFn.layer_order`, not numeric layer order)."""
    return tuple(sorted(site_names))


def _concat_permutation(jax_order: tuple[str, ...], sizes: dict[str, int]) -> np.ndarray:
    """Index array mapping a JAX-order block concatenation to torch (sorted) order:
    `concat_jax[perm] == concat_torch`."""
    offsets: dict[str, int] = {}
    offset = 0
    for site in jax_order:
        offsets[site] = offset
        offset += sizes[site]
    return np.concatenate(
        [np.arange(offsets[s], offsets[s] + sizes[s]) for s in torch_site_order(jax_order)]
    )


def components_state(components: DecompVU, sites: tuple[SiteSpec, ...]) -> dict[str, np.ndarray]:
    """Per-site `model.<site>.components.{V,U}` (fp32)."""
    out: dict[str, np.ndarray] = {}
    for spec in sites:
        V, U = components.site(spec.name)
        out[f"model.{spec.name}.components.V"] = _f32(V)
        out[f"model.{spec.name}.components.U"] = _f32(U)
    return out


def ci_fn_state(ci_fn: CIFn, sites: tuple[SiteSpec, ...]) -> dict[str, np.ndarray]:
    """`CIFn` -> `ci_fn._global_ci_fn.*` (fp32), permuting the in-proj row blocks and
    out-head column blocks from JAX site order to torch sorted order."""
    jax_order = tuple(s.name for s in sites)
    assert jax_order == ci_fn.site_names, (jax_order, ci_fn.site_names)
    assert tuple(s.C for s in sites) == ci_fn.split_sizes
    row_perm = _concat_permutation(jax_order, {s.name: s.d_in for s in sites})
    col_perm = _concat_permutation(jax_order, {s.name: s.C for s in sites})

    out: dict[str, np.ndarray] = {
        f"{CI_FN_PREFIX}._input_projector.W": _f32(ci_fn.in_proj_w)[row_perm, :],
        f"{CI_FN_PREFIX}._input_projector.b": _f32(ci_fn.in_proj_b),
        f"{CI_FN_PREFIX}._output_head.W": _f32(ci_fn.out_w)[:, col_perm],
        f"{CI_FN_PREFIX}._output_head.b": _f32(ci_fn.out_b)[col_perm],
    }
    for i, block in enumerate(ci_fn.blocks):
        prefix = f"{CI_FN_PREFIX}._blocks.{i}"
        out[f"{prefix}.attn.q_proj.weight"] = _f32(block.wq)
        out[f"{prefix}.attn.k_proj.weight"] = _f32(block.wk)
        out[f"{prefix}.attn.v_proj.weight"] = _f32(block.wv)
        out[f"{prefix}.attn.out_proj.weight"] = _f32(block.wo)
        out[f"{prefix}.attn.rope.inv_freq"] = _f32(ci_fn.inv_freq)
        out[f"{prefix}.mlp.0.W"] = _f32(block.w1)
        out[f"{prefix}.mlp.0.b"] = _f32(block.b1)
        out[f"{prefix}.mlp.2.W"] = _f32(block.w2)
        out[f"{prefix}.mlp.2.b"] = _f32(block.b2)
    return out


def frozen_target_keys(n_layer: int, decomposed_sites: frozenset[str]) -> dict[str, str]:
    """`{torch_state_dict_key: hf_safetensors_key}` for the frozen Llama weights. The
    rename is identity-with-`model.`-prefix except: decomposed sites store their frozen
    weight as `.target_weight` (a `ComponentLinear` buffer), and HF's bare `lm_head`
    keeps the `model.` prefix in `LMComponentModel` (the vendored target sits under
    its `model` attribute)."""
    out = {"model.embed_tokens.weight": "model.embed_tokens.weight"}
    for i in range(n_layer):
        prefix = f"model.layers.{i}"
        for sub in ("input_layernorm.weight", "post_attention_layernorm.weight"):
            out[f"{prefix}.{sub}"] = f"{prefix}.{sub}"
        for kind in KIND_ORDER:
            site = site_name(i, kind)
            param = "target_weight" if site in decomposed_sites else "weight"
            out[f"model.{site}.{param}"] = f"model.{site}.weight"
    out["model.norm.weight"] = "model.norm.weight"
    out["model.lm_head.weight"] = "lm_head.weight"
    return out


def frozen_target_state(model_name: str, decomposed_sites: frozenset[str]) -> dict[str, np.ndarray]:
    """The frozen Llama weights under torch's `model.*` keys (fp32, exact bf16 upcast —
    torch's checkpoints carry them because `LMComponentModel.state_dict()` is strict)."""
    hf = _HFWeights(_hf_snapshot_dir(model_name))
    keys = frozen_target_keys(llama31_8b_config().n_layer, decomposed_sites)
    return {torch_key: _f32(hf.get(hf_key)) for torch_key, hf_key in keys.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--step", type=int, default=None, help="checkpoint step (default: latest)")
    args = ap.parse_args()
    jax.config.update("jax_platforms", "cpu")

    cfg = load_run_dir_config(args.run_dir)
    assert isinstance(cfg.target, TargetConfig), (
        f"export implements the llama8b target only, got {type(cfg.target).__name__}"
    )
    lm = llama_decomposed_lm(
        llama31_8b_config(), llama_site_specs(llama31_8b_config(), cfg.target.sites)
    )

    opt_vu, opt_ci, _schedules = build_optimizers(cfg)
    init_key, src_key, _run_key = random.split(random.PRNGKey(cfg.seed), 3)
    reference = init_train_state(cfg, lm, opt_vu, opt_ci, init_key, src_key, dp_mesh())

    manager = make_checkpoint_manager(args.run_dir / "ckpts", cfg.cadence.keep_last)
    step = args.step if args.step is not None else manager.latest_step()
    assert step is not None, f"no checkpoints under {args.run_dir / 'ckpts'}"
    state = restore_step(manager, reference, step)
    assert isinstance(state.components, DecompVU)

    tensors = components_state(state.components, lm.sites)
    tensors |= ci_fn_state(state.ci_fn, lm.sites)
    tensors |= frozen_target_state(cfg.target.model_name, frozenset(lm.site_names))

    out_path = args.run_dir / "export" / f"model_{step}.safetensors"
    out_path.parent.mkdir(exist_ok=True)
    save_file(tensors, str(out_path))
    total_gb = sum(t.nbytes for t in tensors.values()) / 1e9
    print(f"wrote {out_path} ({len(tensors)} tensors, {total_gb:.1f} GB, step {step})")


if __name__ == "__main__":
    main()
