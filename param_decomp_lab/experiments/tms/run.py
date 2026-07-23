"""`pd-tms`: run a TMS (Toy Model of Superposition) parameter decomposition on CPU.

The toy domains live lab-side and call the generic core engine
(`param_decomp.run.run_decomposition_training`) as a library — the core itself carries
zero toy-specific code. A TMS run pretrains its tiny target from scratch in-process (the
Anthropic `mean((|x|-out)^2)` objective), then decomposes it through the same engine the
LM uses, validating via the ground-truth identity-CI metric logged every train-log step.

These toys train in seconds; `pd-tms` runs synchronously on CPU in the main venv (no
SLURM / `param_decomp.run` / CUDA). It mints its own `p-<8hex>` run id (toys do not go through
`pd-lm`).
"""

from pathlib import Path
from typing import Any, Literal

import equinox as eqx
import fire
import jax
import yaml
from jax import random
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from param_decomp import placement
from param_decomp.built_run import LAUNCH_CONFIG_FILENAME, BuiltRun
from param_decomp.components import SiteC
from param_decomp.log import setup_logger
from param_decomp.model import Positionless
from param_decomp.recon import build_loss_terms
from param_decomp.run import run_decomposition_training
from param_decomp.sharding import assert_inline_topology, hsdp_mesh
from param_decomp.train import TrainState
from param_decomp_lab.experiments import toy_uv_eval
from param_decomp_lab.experiments.config import (
    assert_canonical_algorithm_config,
    ci_arch,
    run_instance,
)
from param_decomp_lab.experiments.tms import model as tms
from param_decomp_lab.experiments.tms.config import TMSExperimentConfig
from param_decomp_lab.infra.run_files import generate_run_id


def build_tms_built_run(cfg: TMSExperimentConfig, run_id: str) -> BuiltRun:
    """Convert the canonical TMS schema to the engine's `BuiltRun` bundle via the shared
    helpers. TMS validates via the in-loop target-CI metric (not the LM CEandKLLosses scalar
    pass), so `eval` is `None`. The schema's `eval.metrics` list is still read at run time
    for the config-gated `UVPlots` figure (`toy_uv_eval`)."""
    site_cs = tms.canonical_site_cs(
        tuple(SiteC(s.name, s.C) for s in cfg.decomposition.sites.sites)
    )
    assert_canonical_algorithm_config(cfg)
    build_loss_terms(
        cfg.pd.loss_metrics,
        tuple(sc.name for sc in site_cs),
    )
    target = tms.TMSTargetConfig(
        n_features=cfg.target.n_features,
        n_hidden=cfg.target.n_hidden,
        n_hidden_layers=cfg.target.n_hidden_layers,
        hidden_layer_init=cfg.target.hidden_layer_init,
        init_bias_to_zero=cfg.target.init_bias_to_zero,
        sites=site_cs,
        pretrain_steps=cfg.target.pretrain.steps,
        pretrain_batch_size=cfg.target.pretrain.batch_size,
        pretrain_lr=cfg.target.pretrain.lr,
        pretrain_seed=cfg.target.pretrain.seed,
        feature_probability=cfg.data.feature_probability,
        data_generation_type=cfg.data.data_generation_type,
        global_batch=cfg.pd.batch_size,
    )
    return BuiltRun(
        pd=cfg.pd,
        runtime=cfg.runtime,
        cadence=cfg.cadence,
        run=run_instance(cfg, run_id),
        target=target,
        data=None,
        ci_fn=ci_arch(cfg.decomposition.ci),
        eval=None,
    )


def run_tms_decomposition(built: BuiltRun, raw_cfg: dict[str, Any], mesh: Mesh) -> None:
    """Build + pretrain the TMS target, then decompose it through the generic engine.

    The batch entering the decomposed model IS the raw input `x`. The
    `eval_fn` reads the `lower_leaky` CI of the single-feature probe and logs the
    ground-truth `IdentityCIError` per site every train-log step (TMS has no separate eval
    cadence — `eval_every = cadence.train_log_every`), plus the per-site-permuted CI
    heatmap image alongside each checkpoint (`toy_uv_eval.log_permuted_ci_heatmap` — the
    frozen `hidden_layers.*` sites permute toward dense, not identity)."""
    target_cfg = built.target
    assert isinstance(target_cfg, tms.TMSTargetConfig)
    is_main = jax.process_index() == 0

    tms_cfg = tms.TMSConfig(
        n_features=target_cfg.n_features,
        n_hidden=target_cfg.n_hidden,
        n_hidden_layers=target_cfg.n_hidden_layers,
        hidden_layer_init=target_cfg.hidden_layer_init,
        init_bias_to_zero=target_cfg.init_bias_to_zero,
    )
    if is_main:
        print(f"pretraining TMS target ({target_cfg.pretrain_steps} steps)...", flush=True)
    target = tms.pretrain_tms_target(
        tms_cfg,
        target_cfg.feature_probability,
        target_cfg.data_generation_type,
        target_cfg.pretrain_steps,
        target_cfg.pretrain_batch_size,
        target_cfg.pretrain_lr,
        target_cfg.pretrain_seed,
    )
    # The model IS the frozen target: one `eqx.Module` carries the TMS weights as a field and
    # the decomposition contract as methods.
    model = tms.replicate_target(
        tms.tms_decomposed_model(tms_cfg, target, tms.site_specs(tms_cfg, target_cfg.sites)), mesh
    )

    data_key = random.fold_in(random.PRNGKey(built.pd.seed), 17)

    @jax.jit
    def sample_residual(step_key: jax.Array) -> jax.Array:
        x = tms.sample_sparse_features(
            step_key,
            target_cfg.global_batch,
            target_cfg.n_features,
            target_cfg.feature_probability,
            target_cfg.data_generation_type,
        )
        return jax.lax.with_sharding_constraint(x, NamedSharding(mesh, P(("replicate", "fsdp"))))

    def sample_batch(step: int) -> jax.Array:
        return sample_residual(random.fold_in(data_key, step))

    # `model` is the filter_jit ARG (frozen TMS weights traced, not baked) — closing over an
    # array-bearing eqx model would bake its weights into the HLO.
    @eqx.filter_jit
    def single_feature_ci(
        model: tms.TMSDecomposedModel, ci_fn: Any
    ) -> tuple[dict[str, jax.Array], dict[str, jax.Array]]:
        probe = tms.single_feature_probe(target_cfg.n_features)
        ci = ci_fn(model.read_activations(probe, ci_fn.input_names), remat=False)
        return ci.lower, ci.upper

    uv_spec = toy_uv_eval.toy_uv_spec(model, raw_cfg)
    # The frozen `hidden_layers.*` sites (the `-id` variant) target DENSE recovery — every
    # direction stays live — not identity; canonical torch parity (`tms_40-10-id_config.yaml`
    # `dense_patterns: [hidden_layers.0]`). `linear1`/`linear2` target identity.
    ci_permutation: dict[str, Literal["identity", "dense"]] = {
        site: "dense" if site.startswith("hidden_layers.") else "identity"
        for site in model.site_names
    }

    def eval_fn(state: TrainState, now_step: int) -> dict[str, float]:
        ci_lower, ci_upper = single_feature_ci(model, state.decomposition.ci_fn)
        toy_uv_eval.log_uv_figure(
            uv_spec,
            dict(state.decomposition.components.sites_items()),
            ci_upper,
            now_step,
            wandb_active=built.run.wandb is not None,
        )
        if toy_uv_eval.permuted_ci_heatmap_due(now_step, built.pd.steps, built.cadence.save_every):
            toy_uv_eval.log_permuted_ci_heatmap(
                ci_lower,
                ci_upper,
                ci_permutation,
                now_step,
                wandb_active=built.run.wandb is not None,
            )
        # `identity_ci_error` only applies to identity-target sites (torch parity: the
        # canonical -id configs never scored `hidden_layers.*` numerically, only visually).
        return {
            f"eval/identity_ci_error/{site}": float(tms.identity_ci_error(ci, tolerance=0.1))
            for site, ci in ci_lower.items()
            if ci_permutation[site] == "identity"
        }

    run_decomposition_training(
        pd=built.pd,
        cadence=built.cadence,
        run=built.run,
        model=model,
        ci_fn=built.ci_fn,
        positions=Positionless(),
        remat_recon_forwards=built.runtime.remat_recon_forwards,
        remat_ci_fn=built.runtime.remat_ci_fn,
        ascend_replicate=built.runtime.ascend_replicate,
        compiler_options=built.runtime.compiler_options,
        profile=built.runtime.launch_env.profile,
        sample_batch=sample_batch,
        eval_fn=eval_fn,
        eval_every=built.cadence.train_log_every,
        mesh=mesh,
        placement_rules=placement.from_config(built.runtime.sharding, mesh, model.sites),
    )


def main(config: str, group: str | None = None, tags: str | tuple[str, ...] | None = None) -> None:
    schema_raw = yaml.safe_load(Path(config).read_text())
    run_id = generate_run_id("param_decomp")
    if group is not None or tags is not None:
        wandb_cfg = dict(schema_raw.get("wandb") or {})
        if group is not None:
            wandb_cfg["group"] = group
        if tags is not None:
            # Fire parses a comma-separated `--tags a,b,c` into a tuple, but keeps a value
            # with a hyphen (e.g. `a,b,c-d`) as a string — normalize both to a list.
            wandb_cfg["tags"] = (
                [s.strip() for s in tags.split(",") if s.strip()]
                if isinstance(tags, str)
                else [str(t).strip() for t in tags]
            )
        schema_raw["wandb"] = wandb_cfg
    built = build_tms_built_run(TMSExperimentConfig(**schema_raw), run_id)
    built.run.run_dir.mkdir(parents=True, exist_ok=True)
    setup_logger(built.run.run_dir / "logs.log")
    (built.run.run_dir / LAUNCH_CONFIG_FILENAME).write_text(
        yaml.safe_dump(schema_raw, sort_keys=False)
    )
    assert built.runtime.launch == "inline", "pd-tms is the in-process CPU toy CLI"
    assert_inline_topology(built.runtime.dp)
    mesh = hsdp_mesh()
    run_tms_decomposition(built, schema_raw, mesh)


def cli() -> None:
    fire.Fire(main)


if __name__ == "__main__":
    cli()
