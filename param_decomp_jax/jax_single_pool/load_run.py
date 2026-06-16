"""Open a finished/live JAX single-pool run for offline consumption (harvest, and the
consumers that follow it: clustering, autointerp, slow-eval, app).

This is the reusable "load a JAX run" pattern. It reads a run dir
(`runs/<p-id>/{config.yaml, experiment_config.yaml, ckpts/}`), rebuilds the frozen
target + `DecomposedModel` from the pinned config, restores the orbax checkpoint onto a
reference `TrainState`, and exposes the pure forward a consumer needs:

    run = open_jax_run(run_dir)                 # latest checkpoint
    fwd = run.forward(token_ids)                # one frozen, forward-only pass
    fwd.lower_leaky_ci[site]                    # (B, T, C) leaky CI per site
    fwd.component_acts[site]                    # (B, T, C) ‖U_c‖ · (x @ V) per site
    fwd.output_probs                            # (B, T, vocab) softmax of clean logits

No torch, no `jsp-export` safetensors bridge: the V/U + CI fn come straight from the
orbax checkpoint and the target is built from its own config. CPU-friendly (jax falls
back to CPU); a single device is enough for a small harvest.

`forward` mirrors the forward-only subset of `eval.make_eval_step`: clean logits +
`site_inputs` + lower-leaky CI, plus per-component acts (the harvest extra). bf16
compute on the components / CI fn (training's `COMPUTE_DT`) so consumed CI matches the
trained model's; output probs are fp32 from the fp32-upcast frozen forward.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from jax_single_pool import llama_simple_mlp
from jax_single_pool.checkpoint import make_checkpoint_manager, restore_latest, restore_step
from jax_single_pool.config import (
    ExperimentConfig,
    LlamaSimpleMLPTargetConfig,
    TargetConfig,
    load_run_dir_config,
)
from jax_single_pool.llama8b import DecompVU
from jax_single_pool.lm import DecomposedModel
from jax_single_pool.run_state import build_optimizers, init_train_state
from jax_single_pool.sharding import dp_mesh
from jax_single_pool.train import COMPUTE_DT, TrainState, cast_floating


@dataclass(frozen=True)
class HarvestForward:
    """One frozen forward-only pass over a token batch, the raw material every harvest
    fn turns into per-component statistics. Site-name-keyed; `(B, T, C_site)`."""

    lower_leaky_ci: dict[str, Float[Array, "B T C"]]
    component_acts: dict[str, Float[Array, "B T C"]]
    output_probs: Float[Array, "B T vocab"]


def _build_target(cfg: ExperimentConfig, mesh: jax.sharding.Mesh):
    """Frozen target + prefix-residual fn for the run's target config. SimpleMLP reads
    its local pretrain cache (no network); llama8b reads the HF snapshot."""
    match cfg.target:
        case LlamaSimpleMLPTargetConfig():
            cache_dir = llama_simple_mlp.pretrain_cache_dir(cfg.target.pretrain_run_path)
            simple_cfg = llama_simple_mlp.load_model_config(cache_dir)
            lm = llama_simple_mlp.llama_simple_mlp_decomposed_lm(
                simple_cfg, llama_simple_mlp.site_specs(simple_cfg, cfg.target.sites)
            )
            first_layer = llama_simple_mlp.first_decomposed_layer(lm.site_names)
            target = llama_simple_mlp.replicate_frozen(
                llama_simple_mlp.load_target_from_pretrain_cache(
                    cache_dir, simple_cfg, first_layer, jnp.bfloat16
                ),
                mesh,
            )
            prefix = llama_simple_mlp.replicate_frozen(
                llama_simple_mlp.load_prefix_from_pretrain_cache(
                    cache_dir, simple_cfg, first_layer, jnp.bfloat16
                ),
                mesh,
            )
            return lm, target, prefix, llama_simple_mlp.prefix_residual, simple_cfg.vocab_size
        case TargetConfig():
            raise NotImplementedError(
                "open_jax_run currently builds the LlamaSimpleMLP target only; the "
                "llama8b target needs HF weight loading + sharding (see run.py main)."
            )


def _u_norms(components: DecompVU, site_names: tuple[str, ...]) -> dict[str, Float[Array, " C"]]:
    """Per-component output-direction magnitude ‖U_c‖ — the harvest `component_activation`
    scale (torch `harvest_fn/param_decomp.py`: `component.U.norm(dim=1)`)."""
    return {
        site: jnp.linalg.norm(components.site(site)[1].astype(jnp.float32), axis=1)
        for site in site_names
    }


@dataclass(frozen=True)
class LoadedJaxRun:
    """A JAX run opened for consumption: restored trajectory + frozen target + the pure
    forward consumers need. `layer_activation_sizes` / `vocab_size` mirror the torch
    `PDAdapter` fields the harvest pipeline keys on."""

    run_id: str
    step: int
    lm: DecomposedModel
    config: ExperimentConfig
    vocab_size: int
    _state: TrainState
    _forward: Any

    @property
    def site_names(self) -> tuple[str, ...]:
        return self.lm.site_names

    @property
    def layer_activation_sizes(self) -> list[tuple[str, int]]:
        """`(site_name, C)` per decomposed site, in canonical order — the harvest
        accumulator's `layers` argument."""
        return [(s.name, s.C) for s in self.lm.sites]

    def forward(self, token_ids: Int[Array, "B T"]) -> HarvestForward:
        lower_leaky_ci, component_acts, output_probs = self._forward(
            self._state.components, self._state.ci_fn, token_ids
        )
        return HarvestForward(
            lower_leaky_ci=lower_leaky_ci,
            component_acts=component_acts,
            output_probs=output_probs,
        )


def open_jax_run(run_dir: Path, step: int | None = None) -> LoadedJaxRun:
    """Open the run at `run_dir`; restore checkpoint `step` (latest if None)."""
    cfg = load_run_dir_config(run_dir)
    mesh = dp_mesh()
    lm, target, prefix, prefix_residual_fn, vocab_size = _build_target(cfg, mesh)

    opt_vu, opt_ci, _ = build_optimizers(cfg)
    init_key, src_key = jax.random.split(jax.random.PRNGKey(cfg.seed))
    reference = init_train_state(cfg, lm, opt_vu, opt_ci, init_key, src_key, mesh)

    manager = make_checkpoint_manager(run_dir / "ckpts", cfg.cadence.keep_last)
    if step is None:
        restored = restore_latest(manager, reference)
        assert restored is not None, f"no checkpoints under {run_dir / 'ckpts'}"
        state, resolved_step = restored
    else:
        state, resolved_step = restore_step(manager, reference, step), step
    assert isinstance(state.components, DecompVU)

    site_names = lm.site_names
    u_norms = _u_norms(state.components, site_names)

    @jax.jit
    def forward(
        components: DecompVU, ci_fn: Any, token_ids: Int[Array, "B T"]
    ) -> tuple[dict[str, Array], dict[str, Array], Array]:
        residual = prefix_residual_fn(prefix, token_ids)
        clean_output = lm.clean_output(target, residual)
        site_inputs = lm.site_inputs(target, residual)

        components_bf16 = cast_floating(components, COMPUTE_DT)
        ci_fn_bf16 = cast_floating(ci_fn, COMPUTE_DT)
        lower_ci = ci_fn_bf16(site_inputs).lower

        component_acts = {}
        for site in site_names:
            V = components_bf16.site(site)[0]
            acts = site_inputs[site].astype(COMPUTE_DT) @ V  # (B, T, C): x @ V
            component_acts[site] = acts.astype(jnp.float32) * u_norms[site]

        return (
            {site: lower_ci[site].astype(jnp.float32) for site in site_names},
            component_acts,
            jax.nn.softmax(clean_output.astype(jnp.float32), axis=-1),
        )

    return LoadedJaxRun(
        run_id=cfg.run_id,
        step=resolved_step,
        lm=lm,
        config=cfg,
        vocab_size=vocab_size,
        _state=state,
        _forward=forward,
    )
