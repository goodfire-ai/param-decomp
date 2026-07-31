"""Open a finished/live JAX single-pool run for offline consumption (harvest, and the
consumers that follow it: clustering, autointerp, slow-eval, app).

This is the reusable "load a JAX run" pattern. It reads a run dir
(`runs/<p-id>/{launch_config.yaml, ckpts/}`), rebuilds the frozen
target + `DecomposedModel` from the pinned config, restores the checkpoint's
`decomposition` item (the trained V/U + ci_fn — optimizer/adversary state is training's
business and is never touched), and exposes the pure forward a consumer needs:

    run = open_jax_run(run_dir, data_root=data_root)   # latest checkpoint
    fwd = run.forward(token_ids)                # one frozen, forward-only pass
    fwd.lower_leaky_ci[site]                    # (B, T, C) leaky CI per site
    fwd.component_acts[site]                    # (B, T, C) ‖U_c‖ · (x @ V) per site
    fwd.output_probs                            # (B, T, vocab) softmax of clean logits

No torch, no safetensors bridge: the V/U + CI fn come straight from the
orbax checkpoint and the target is built from its own config. CPU-friendly (jax falls
back to CPU); a single device is enough for a small harvest.

`forward` mirrors the forward-only subset of `eval.make_eval_step`: clean logits +
the CI fn's residual taps + lower-leaky CI, plus per-component acts (the harvest extra,
from the frozen per-site matrix inputs `model.read_activations` serves for site-name keys). bf16
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

from param_decomp.core import placement
from param_decomp.core.built_run import LAUNCH_CONFIG_FILENAME
from param_decomp.core.checkpoint import (
    make_read_only_checkpoint_manager,
    restore_decomposition_to_host,
)
from param_decomp.core.ci_fn import ChunkwiseTransformerCIFn
from param_decomp.core.components import ComponentStacks
from param_decomp.core.configs import PlacementTableConfig
from param_decomp.core.model import DecomposedModel
from param_decomp.core.run_state import init_decomposition
from param_decomp.core.sharding import hsdp_mesh, place_target, place_via_shardings
from param_decomp.core.train import COMPUTE_DT, Decomposition, cast_floating
from param_decomp.experiments.lm.config import (
    LlamaSimpleMLPTargetConfig,
    TargetConfig,
    hf_model_family,
    load_config,
)
from param_decomp.experiments.lm.resolved import LMRun, weights_jnp_dtype
from param_decomp.infra import pretrain_cache
from param_decomp.targets import llama_simple_mlp
from param_decomp.targets.glu_transformer import glu_site_specs


@dataclass(frozen=True)
class HarvestForward:
    """One frozen forward-only pass over a token batch, the raw material every harvest
    fn turns into per-component statistics. Site-name-keyed; `(B, T, C_site)`."""

    lower_leaky_ci: dict[str, Float[Array, "B T C"]]
    component_acts: dict[str, Float[Array, "B T C"]]
    output_probs: Float[Array, "B T vocab"]


def build_target(
    cfg: LMRun, mesh: jax.sharding.Mesh, data_root: Path
) -> tuple[DecomposedModel, int]:
    """`(model, vocab_size)` for the run's target config. The `model` (an `eqx.Module`) IS the
    frozen target — it carries the full model weights (embedding included) as fields and
    embeds its token input internally. SimpleMLP reads its pretrain cache under `data_root`
    (no network); the HF families read the HF snapshot. Both cast their weights to the
    config's `target.weights_dtype` on read — this is the ONLY place that dtype is applied,
    so train and consume load the same target.

    LM-only: harvest/slow-eval over the toy (TMS/ResidMLP) targets is not wired — those
    validate via their in-loop target-CI metric in the lab provider, not this path."""
    match cfg.target:
        case LlamaSimpleMLPTargetConfig():
            cache_dir = pretrain_cache.resolved_cache_dir(data_root, cfg.target.pretrain_run_path)
            simple_cfg = llama_simple_mlp.load_model_config(cache_dir)
            sites = llama_simple_mlp.site_specs(simple_cfg, cfg.target.sites)
            loaded_model = llama_simple_mlp.load_decomposed_lm_from_pretrain_cache(
                cache_dir, simple_cfg, sites, weights_jnp_dtype(cfg.target.weights_dtype)
            )
            model = place_via_shardings(loaded_model, loaded_model.shardings(mesh))
            return model, simple_cfg.vocab_size
        case TargetConfig():
            family = hf_model_family(cfg.target.model_name)
            arch_cfg = family.arch_config()
            sites = glu_site_specs(arch_cfg, cfg.target.sites)
            loaded_model = family.load(
                cfg.target.model_name,
                arch_cfg,
                sites,
                weights_jnp_dtype(cfg.target.weights_dtype),
            )
            return place_target(loaded_model, mesh), arch_cfg.vocab_size
        case _:
            raise AssertionError(f"build_target is LM-only; got target {type(cfg.target).__name__}")


def _u_norms(
    components: ComponentStacks, site_names: tuple[str, ...]
) -> dict[str, Float[Array, " C"]]:
    """Per-component output-direction magnitude ‖U_c‖ — the harvest `component_activation`
    scale (torch `harvest_fn/param_decomp.core.py`: `component.U.norm(dim=1)`)."""
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
    model: DecomposedModel
    config: LMRun
    vocab_size: int
    _decomposition: Decomposition
    _forward: Callable[
        [DecomposedModel, ComponentStacks, ChunkwiseTransformerCIFn, Int[Array, "B T"]],
        tuple[dict[str, Array], dict[str, Array], Array],
    ]

    @property
    def site_names(self) -> tuple[str, ...]:
        return self.model.site_names

    @property
    def layer_activation_sizes(self) -> list[tuple[str, int]]:
        """`(site_name, C)` per decomposed site, in canonical order — the harvest
        accumulator's `layers` argument."""
        return [(s.name, s.C) for s in self.model.sites]

    def forward(self, token_ids: Int[Array, "B T"]) -> HarvestForward:
        ci_fn = self._decomposition.ci_fn
        assert isinstance(ci_fn, ChunkwiseTransformerCIFn), (
            "harvest is the transformer-CI-fn (LM) path only"
        )
        lower_leaky_ci, component_acts, output_probs = self._forward(
            self.model, self._decomposition.components, ci_fn, token_ids
        )
        return HarvestForward(
            lower_leaky_ci=lower_leaky_ci,
            component_acts=component_acts,
            output_probs=output_probs,
        )


def _restore_decomposition(
    cfg: LMRun,
    sharding: str | PlacementTableConfig,
    model: DecomposedModel,
    mesh: jax.sharding.Mesh,
    run_dir: Path,
    step: int | None,
) -> tuple[Decomposition, int]:
    """Restore ONLY the trained decomposition from the run's checkpoint.

    Consumers never need the optimizer moments or persistent-PGD adversary sources
    training also checkpoints — and on a single device those dominate: an 8B run's
    sources + Adam state materialize ~60GB at init and again at restore staging, which
    wedges/OOMs one GPU. `jax.eval_shape` over `init_decomposition` yields the saved
    decomposition's structure with ZERO allocation (and zero knowledge of training's
    optimizers); leaves restore as host numpy, then `device_put` onto the consumer's
    single default device."""
    init_key, _ = jax.random.split(jax.random.PRNGKey(cfg.pd.seed))
    # Consumer construction: this mesh is the CONSUMER's topology (often one device), not
    # the run's launch topology, so the launch claims don't bind — a zero1 row declared
    # for dp=N is legitimately unreachable here.
    rules = placement.from_config_for_consumer(sharding, mesh, model.sites)
    abstract = jax.eval_shape(lambda: init_decomposition(model, cfg.ci_fn, init_key, mesh, rules))

    manager = make_read_only_checkpoint_manager(run_dir / "ckpts")
    resolved_step = manager.latest_step() if step is None else step
    assert resolved_step is not None, f"no checkpoints under {run_dir / 'ckpts'}"
    decomposition = jax.device_put(restore_decomposition_to_host(manager, resolved_step, abstract))
    return decomposition, resolved_step


def open_jax_run(run_dir: Path, step: int | None = None, *, data_root: Path) -> LoadedJaxRun:
    """Open the run at `run_dir`; restore checkpoint `step` (latest if None). Restores
    only the trained decomposition (see `_restore_decomposition`). `data_root` resolves a
    `kind: pretrained` target's cache (`<data_root>/pretrain_cache/...`)."""
    cfg, authored = load_config(run_dir / LAUNCH_CONFIG_FILENAME, run_dir.name, data_root)
    mesh = hsdp_mesh()
    target_model, vocab_size = build_target(cfg, mesh, data_root)
    decomposition, resolved_step = _restore_decomposition(
        cfg, authored.runtime.sharding, target_model, mesh, run_dir, step
    )
    assert isinstance(decomposition.components, ComponentStacks)

    site_names = target_model.site_names
    u_norms = _u_norms(decomposition.components, site_names)

    # `model` is the filter_jit ARG (frozen weights traced, not baked; distinct from the
    # outer `target_model` so an accidental closure capture fails loudly). It embeds the
    # token ids internally — the harvest forward feeds tokens straight in.
    @eqx.filter_jit
    def forward(
        model: DecomposedModel,
        components: ComponentStacks,
        ci_fn: ChunkwiseTransformerCIFn,
        token_ids: Int[Array, "B T"],
    ) -> tuple[dict[str, Array], dict[str, Array], Array]:
        clean_output, all_taps = model.clean_output_and_activations(
            token_ids, ci_fn.input_names + site_names
        )
        taps = {k: all_taps[k] for k in ci_fn.input_names}
        site_inputs = {k: all_taps[k] for k in site_names}

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
        model=target_model,
        config=cfg,
        vocab_size=vocab_size,
        _decomposition=decomposition,
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


def run_metadata(run_dir: Path, *, data_root: Path) -> RunMetadata:
    """Target topology for `run_dir`, derived from the pinned config (+ the SimpleMLP
    pretrain cache's `model_config.yaml` for `n_layer`/`vocab_size`). No orbax restore."""
    cfg, _ = load_config(run_dir / LAUNCH_CONFIG_FILENAME, run_dir.name, data_root)
    match cfg.target:
        case LlamaSimpleMLPTargetConfig():
            cache_dir = pretrain_cache.resolved_cache_dir(data_root, cfg.target.pretrain_run_path)
            simple_cfg = llama_simple_mlp.load_model_config(cache_dir)
            return RunMetadata(
                model_type="LlamaSimpleMLP",
                n_blocks=simple_cfg.n_layer,
                vocab_size=simple_cfg.vocab_size,
                layer_activation_sizes=[(s.name, s.C) for s in cfg.target.sites],
            )
        case TargetConfig():
            family = hf_model_family(cfg.target.model_name)
            arch_cfg = family.arch_config()
            return RunMetadata(
                model_type=family.model_type,
                n_blocks=arch_cfg.n_layer,
                vocab_size=arch_cfg.vocab_size,
                layer_activation_sizes=[(s.name, s.C) for s in cfg.target.sites],
            )
        case _:
            raise AssertionError(f"run_metadata is LM-only; got target {type(cfg.target).__name__}")
