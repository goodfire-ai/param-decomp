"""`pd-resid-mlp`: run a ResidualMLP parameter decomposition on CPU.

The SPD/APD residual-stream toy lives lab-side and calls the generic core engine
(`param_decomp.run.run_decomposition_training`) as a library. The target pretrains from
scratch in-process (the `act_fn(coeffs·x) + x` read-off objective), then decomposes through
the same engine the LM uses, validating via the ground-truth identity-CI metric.

These toys train in seconds; `pd-resid-mlp` runs synchronously on CPU in the main venv
(no SLURM / `param_decomp.run` / CUDA). It mints its own `p-<8hex>` run id.
"""

from pathlib import Path
from typing import Any, Literal

import equinox as eqx
import fire
import jax
import numpy as np
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
from param_decomp.sharding import hsdp_mesh
from param_decomp.slow_eval import dense_ci_error
from param_decomp.train import TrainState
from param_decomp_lab.experiments import toy_uv_eval
from param_decomp_lab.experiments.config import (
    assert_canonical_algorithm_config,
    ci_arch,
    run_instance,
)
from param_decomp_lab.experiments.resid_mlp import model as resid_mlp
from param_decomp_lab.experiments.resid_mlp.config import ResidMLPExperimentConfig
from param_decomp_lab.infra.run_files import generate_run_id


def build_resid_mlp_built_run(cfg: ResidMLPExperimentConfig, run_id: str) -> BuiltRun:
    """Convert the canonical ResidMLP schema to the engine's `BuiltRun` bundle via the shared
    helpers. ResidMLP validates via the in-loop target-CI metric (not the LM CEandKLLosses
    scalar pass), so `eval` is `None`. The schema's `eval.metrics` list is still read at run
    time for the config-gated `UVPlots` figure (`toy_uv_eval`)."""
    site_cs = resid_mlp.canonical_site_cs(
        tuple(SiteC(s.name, s.C) for s in cfg.decomposition.sites.sites)
    )
    assert_canonical_algorithm_config(cfg)
    build_loss_terms(
        cfg.pd.loss_metrics,
        tuple(sc.name for sc in site_cs),
    )
    target = resid_mlp.ResidMLPTargetConfig(
        n_features=cfg.target.n_features,
        d_embed=cfg.target.d_embed,
        d_mlp=cfg.target.d_mlp,
        n_layers=cfg.target.n_layers,
        act_fn_name=cfg.target.act_fn_name,
        in_bias=cfg.target.in_bias,
        out_bias=cfg.target.out_bias,
        fixed_identity_embedding=cfg.target.fixed_identity_embedding,
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


def run_resid_mlp_decomposition(built: BuiltRun, raw_cfg: dict[str, Any], mesh: Mesh) -> None:
    """Build + pretrain the ResidMLP target, then decompose it through the generic engine.

    The batch entering the decomposed model is `x @ W_E` (`W_E` is carried inside the
    frozen target, not decomposed). The `eval_fn` reads the `lower_leaky` CI of the
    single-feature probe (embedded through `W_E`) and logs the ground-truth `IdentityCIError`
    per site every train-log step (`eval_every = cadence.train_log_every`), plus the
    per-site-permuted CI heatmap image alongside each checkpoint
    (`toy_uv_eval.log_permuted_ci_heatmap` — `mlp_out` permutes toward dense, not identity)."""
    target_cfg = built.target
    assert isinstance(target_cfg, resid_mlp.ResidMLPTargetConfig)
    is_main = jax.process_index() == 0

    resid_cfg = resid_mlp.ResidMLPConfig(
        n_features=target_cfg.n_features,
        d_embed=target_cfg.d_embed,
        d_mlp=target_cfg.d_mlp,
        n_layers=target_cfg.n_layers,
        act_fn_name=target_cfg.act_fn_name,
        in_bias=target_cfg.in_bias,
        out_bias=target_cfg.out_bias,
        fixed_identity_embedding=target_cfg.fixed_identity_embedding,
    )
    if is_main:
        print(f"pretraining ResidMLP target ({target_cfg.pretrain_steps} steps)...", flush=True)
    target = resid_mlp.pretrain_resid_mlp_target(
        resid_cfg,
        target_cfg.feature_probability,
        target_cfg.data_generation_type,
        target_cfg.pretrain_steps,
        target_cfg.pretrain_batch_size,
        target_cfg.pretrain_lr,
        target_cfg.pretrain_seed,
    )
    # The model IS the frozen target: one `eqx.Module` carries the ResidMLP weights as a field
    # and the decomposition contract as methods.
    model = resid_mlp.replicate_target(
        resid_mlp.resid_mlp_decomposed_model(
            resid_cfg, target, resid_mlp.site_specs(resid_cfg, target_cfg.sites)
        ),
        mesh,
    )

    data_key = random.fold_in(random.PRNGKey(built.pd.seed), 17)

    # `tgt` is the filter_jit ARG (frozen `W_E` traced, not baked) — closing over an
    # array-bearing eqx target would bake its weights into the HLO.
    @eqx.filter_jit
    def sample_residual(tgt: resid_mlp.ResidMLPTarget, step_key: jax.Array) -> jax.Array:
        x = resid_mlp.sample_sparse_features(
            step_key,
            target_cfg.global_batch,
            target_cfg.n_features,
            target_cfg.feature_probability,
            target_cfg.data_generation_type,
        )
        residual = resid_mlp.resid_mlp_input_residual(tgt, x)
        return jax.lax.with_sharding_constraint(
            residual, NamedSharding(mesh, P(("replicate", "fsdp")))
        )

    def sample_batch(step: int) -> jax.Array:
        return sample_residual(model.target, random.fold_in(data_key, step))

    # `model` is the filter_jit ARG (frozen weights traced, not baked).
    @eqx.filter_jit
    def single_feature_ci(
        model: resid_mlp.ResidMLPDecomposedModel, ci_fn: Any
    ) -> tuple[dict[str, jax.Array], dict[str, jax.Array]]:
        resid = resid_mlp.single_feature_probe(target_cfg.n_features) @ model.target.W_E
        ci = ci_fn(model.read_activations(resid, ci_fn.input_names), remat=False)
        return ci.lower, ci.upper

    uv_spec = toy_uv_eval.toy_uv_spec(model, raw_cfg)
    # `mlp_out` targets DENSE recovery (every d_mlp direction stays live), not identity —
    # torch parity (`resid_mlp{1,2,3}_config.yaml` `dense_patterns: [layers.*.mlp_out]`).
    # `mlp_in` targets identity.
    ci_permutation: dict[str, Literal["identity", "dense"]] = {
        site: "dense" if site.endswith(resid_mlp.MLP_OUT) else "identity"
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
        metrics = {
            f"eval/identity_ci_error/{site}": float(resid_mlp.identity_ci_error(ci, tolerance=0.1))
            for site, ci in ci_lower.items()
            if ci_permutation[site] == "identity"
        }
        # `mlp_out`'s target is every d_mlp direction alive (k = the site's own width) —
        # scored separately since it's a different target shape than `identity_ci_error`.
        metrics.update(
            {
                f"eval/dense_ci_error/{site}": float(
                    dense_ci_error(np.asarray(ci_lower[site]), k=target_cfg.d_mlp, tolerance=0.1)
                )
                for site in ci_lower
                if ci_permutation[site] == "dense"
            }
        )
        return metrics

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
    built = build_resid_mlp_built_run(ResidMLPExperimentConfig(**schema_raw), run_id)
    built.run.run_dir.mkdir(parents=True, exist_ok=True)
    setup_logger(built.run.run_dir / "logs.log")
    (built.run.run_dir / LAUNCH_CONFIG_FILENAME).write_text(
        yaml.safe_dump(schema_raw, sort_keys=False)
    )
    mesh = hsdp_mesh()
    run_resid_mlp_decomposition(built, schema_raw, mesh)


def cli() -> None:
    fire.Fire(main)


if __name__ == "__main__":
    cli()
