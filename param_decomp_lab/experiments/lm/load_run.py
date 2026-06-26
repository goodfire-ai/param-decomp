"""Open a finished/live JAX single-pool run for offline consumption (harvest, and the
consumers that follow it: clustering, autointerp, slow-eval, app).

This is the reusable "load a JAX run" pattern. It reads a run dir
(`runs/<p-id>/{config.yaml, ckpts/}`), rebuilds the frozen
target + `DecomposedModel` from the pinned config, restores the orbax checkpoint onto a
reference `TrainState`, and exposes the pure forward a consumer needs:

    run = open_jax_run(run_dir)                 # latest checkpoint
    fwd = run.forward(token_ids)                # one frozen, forward-only pass
    fwd.lower_leaky_ci[site]                    # (B, T, C) leaky CI per site
    fwd.component_acts[site]                    # (B, T, C) ‖U_c‖ · (x @ V) per site
    fwd.output_probs                            # (B, T, vocab) softmax of clean logits

No torch, no safetensors bridge: the V/U + CI fn come straight from the
orbax checkpoint and the target is built from its own config. CPU-friendly (jax falls
back to CPU); a single device is enough for a small harvest.

`forward` mirrors the forward-only subset of `eval.make_eval_step`: clean logits +
the CI fn's residual taps + lower-leaky CI, plus per-component acts (the harvest extra,
from the frozen per-site matrix inputs `lm.read_activations` serves for site-name keys). bf16
compute on the components / CI fn (training's `COMPUTE_DT`) so consumed CI matches the
trained model's; output probs are fp32 from the fp32-upcast frozen forward.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from param_decomp.built_run import BuiltRun
from param_decomp.checkpoint import make_checkpoint_manager, restore_latest, restore_step
from param_decomp.ci_fn import CIFn
from param_decomp.components import DecompVU
from param_decomp.lm import DecomposedModel
from param_decomp.run_state import build_optimizers, init_train_state
from param_decomp.sharding import dp_mesh, place_via_shardings
from param_decomp.targets import llama_simple_mlp
from param_decomp.targets.llama8b import (
    llama31_8b_config,
    llama_site_specs,
    load_decomposed_lm_from_hf,
)
from param_decomp.targets.llama8b_sharding import place_target
from param_decomp.train import COMPUTE_DT, TrainState, cast_floating
from param_decomp_lab.experiments.lm.config import (
    LlamaSimpleMLPTargetConfig,
    TargetConfig,
    load_run_dir_config,
)


@dataclass(frozen=True)
class HarvestForward:
    """One frozen forward-only pass over a token batch, the raw material every harvest
    fn turns into per-component statistics. Site-name-keyed; `(B, T, C_site)`."""

    lower_leaky_ci: dict[str, Float[Array, "B T C"]]
    component_acts: dict[str, Float[Array, "B T C"]]
    output_probs: Float[Array, "B T vocab"]


def build_target(cfg: BuiltRun, mesh: jax.sharding.Mesh) -> tuple[DecomposedModel, int]:
    """`(lm, vocab_size)` for the run's target config. The `lm` (an `eqx.Module`) IS the
    frozen target — it carries the full model weights (embedding included) as fields and
    embeds its token input internally. SimpleMLP reads its local pretrain cache (no network);
    llama8b reads the HF snapshot (frozen bf16 weights + fp32-compute, matching `run.py::main`).

    LM-only: harvest/slow-eval over the toy (TMS/ResidMLP) targets is not wired — those
    validate via their in-loop target-CI metric in the lab provider, not this path."""
    match cfg.target:
        case LlamaSimpleMLPTargetConfig():
            cache_dir = llama_simple_mlp.pretrain_cache_dir(cfg.target.pretrain_run_path)
            simple_cfg = llama_simple_mlp.load_model_config(cache_dir)
            sites = llama_simple_mlp.site_specs(simple_cfg, cfg.target.sites)
            loaded_lm = llama_simple_mlp.load_decomposed_lm_from_pretrain_cache(
                cache_dir, simple_cfg, sites, jnp.bfloat16
            )
            lm = place_via_shardings(loaded_lm, loaded_lm.shardings(mesh))
            return lm, simple_cfg.vocab_size
        case TargetConfig():
            llama_cfg = llama31_8b_config()
            sites = llama_site_specs(llama_cfg, cfg.target.sites)
            lm = place_target(
                load_decomposed_lm_from_hf(cfg.target.model_name, llama_cfg, sites), mesh
            )
            return lm, llama_cfg.vocab_size
        case _:
            raise AssertionError(f"build_target is LM-only; got target {type(cfg.target).__name__}")


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
    config: BuiltRun
    vocab_size: int
    _state: TrainState
    _forward: Callable[
        [DecomposedModel, DecompVU, CIFn, Int[Array, "B T"]],
        tuple[dict[str, Array], dict[str, Array], Array],
    ]

    @property
    def site_names(self) -> tuple[str, ...]:
        return self.lm.site_names

    @property
    def layer_activation_sizes(self) -> list[tuple[str, int]]:
        """`(site_name, C)` per decomposed site, in canonical order — the harvest
        accumulator's `layers` argument."""
        return [(s.name, s.C) for s in self.lm.sites]

    def forward(self, token_ids: Int[Array, "B T"]) -> HarvestForward:
        ci_fn = self._state.ci_fn
        assert isinstance(ci_fn, CIFn), "harvest is the transformer-CI-fn (LM) path only"
        lower_leaky_ci, component_acts, output_probs = self._forward(
            self.lm, self._state.components, ci_fn, token_ids
        )
        return HarvestForward(
            lower_leaky_ci=lower_leaky_ci,
            component_acts=component_acts,
            output_probs=output_probs,
        )


def open_jax_run(run_dir: Path, step: int | None = None) -> LoadedJaxRun:
    """Open the run at `run_dir`; restore checkpoint `step` (latest if None)."""
    cfg = load_run_dir_config(run_dir)
    mesh = dp_mesh(cfg.runtime.tp)
    lm, vocab_size = build_target(cfg, mesh)

    opt_vu, opt_ci, _ = build_optimizers(cfg.pd)
    init_key, src_key = jax.random.split(jax.random.PRNGKey(cfg.pd.seed))
    reference = init_train_state(
        cfg.pd, lm, cfg.ci_fn, cfg.data, opt_vu, opt_ci, init_key, src_key, mesh
    )

    assert cfg.cadence.keep_last_n_checkpoints is not None, cfg.cadence
    manager = make_checkpoint_manager(run_dir / "ckpts", cfg.cadence.keep_last_n_checkpoints)
    if step is None:
        restored = restore_latest(manager, reference)
        assert restored is not None, f"no checkpoints under {run_dir / 'ckpts'}"
        state, resolved_step = restored
    else:
        state, resolved_step = restore_step(manager, reference, step), step
    assert isinstance(state.components, DecompVU)

    site_names = lm.site_names
    u_norms = _u_norms(state.components, site_names)

    # `model` is the filter_jit ARG (frozen weights traced, not baked). It embeds the token
    # ids internally — the harvest forward feeds tokens straight in, no prefix.
    @eqx.filter_jit
    def forward(
        model: DecomposedModel,
        components: DecompVU,
        ci_fn: CIFn,
        token_ids: Int[Array, "B T"],
    ) -> tuple[dict[str, Array], dict[str, Array], Array]:
        clean_output = model.clean_output(token_ids)
        taps = model.read_activations(token_ids, ci_fn.input_names)
        site_inputs = model.read_activations(token_ids, site_names)

        components_bf16 = cast_floating(components, COMPUTE_DT)
        ci_fn_bf16 = cast_floating(ci_fn, COMPUTE_DT)
        lower_ci = ci_fn_bf16(taps, remat=False).lower

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
        run_id=run_dir.name,
        step=resolved_step,
        lm=lm,
        config=cfg,
        vocab_size=vocab_size,
        _state=state,
        _forward=forward,
    )


@dataclass(frozen=True)
class RunMetadata:
    """A JAX run's target topology, read from config + cache WITHOUT restoring a
    checkpoint — the metadata the autointerp/clustering consumers need (`n_blocks`,
    `vocab_size`, per-site `(name, C)`). `model_type` selects the canonical-path schema
    consumers use to render human-readable layer descriptions."""

    model_type: str
    n_blocks: int
    vocab_size: int
    layer_activation_sizes: list[tuple[str, int]]


def run_metadata(run_dir: Path) -> RunMetadata:
    """Target topology for `run_dir`, derived from the pinned config (+ the SimpleMLP
    pretrain cache's `model_config.yaml` for `n_layer`/`vocab_size`). No orbax restore."""
    cfg = load_run_dir_config(run_dir)
    match cfg.target:
        case LlamaSimpleMLPTargetConfig():
            cache_dir = llama_simple_mlp.pretrain_cache_dir(cfg.target.pretrain_run_path)
            simple_cfg = llama_simple_mlp.load_model_config(cache_dir)
            return RunMetadata(
                model_type="LlamaSimpleMLP",
                n_blocks=simple_cfg.n_layer,
                vocab_size=simple_cfg.vocab_size,
                layer_activation_sizes=[(s.name, s.C) for s in cfg.target.sites],
            )
        case TargetConfig():
            llama_cfg = llama31_8b_config()
            return RunMetadata(
                model_type="Llama",
                n_blocks=llama_cfg.n_layer,
                vocab_size=llama_cfg.vocab_size,
                layer_activation_sizes=[(s.name, s.C) for s in cfg.target.sites],
            )
        case _:
            raise AssertionError(
                "run_metadata is the LM-consumer path only (toys are not harvested); "
                f"got target {type(cfg.target).__name__}"
            )
