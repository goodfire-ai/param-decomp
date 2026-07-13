"""`pd-deep-linear`: decompose a deep-linear identity model in-process (CPU or 1 GPU).

Mirrors `pd-tms`, minus pretraining: the target is CONSTRUCTED (`n_layers` frozen
`eye(n_features)` sites), the data is uniform one-hot rows sampled on the fly, and the
recon comparison is the LM's KL-on-final-logits — which makes this the smallest target
that exercises `ChunkwiseSubsetReconLoss` end-to-end. Validated via the ground-truth
identity-CI metric (the TMS `identity_ci_error`, probe magnitude 1.0) every
train-log step.
"""

from pathlib import Path
from typing import Any

import equinox as eqx
import fire
import jax
import yaml
from jax import random
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from param_decomp.built_run import LAUNCH_CONFIG_FILENAME, BuiltRun
from param_decomp.components import SiteC
from param_decomp.log import setup_logger
from param_decomp.recon import build_loss_terms
from param_decomp.run import run_decomposition_training
from param_decomp.sharding import hsdp_mesh
from param_decomp.train import TrainState
from param_decomp_lab.experiments import toy_uv_eval
from param_decomp_lab.experiments.config import (
    assert_canonical_algorithm_config,
    ci_arch,
    run_instance,
)
from param_decomp_lab.experiments.deep_linear import model as deep_linear
from param_decomp_lab.experiments.deep_linear.config import DeepLinearExperimentConfig
from param_decomp_lab.experiments.tms.model import identity_ci_error
from param_decomp_lab.infra.run_files import generate_run_id


def build_deep_linear_built_run(cfg: DeepLinearExperimentConfig, run_id: str) -> BuiltRun:
    """Convert the canonical schema to the engine's `BuiltRun` bundle via the shared
    helpers. Like the other toys, validation is the in-loop target-CI metric, so the
    core `eval` stays `None`."""
    site_cs = deep_linear.expand_wildcard_site_cs(
        tuple(SiteC(t.module_pattern, t.C) for t in cfg.pd.decomposition_targets),
        cfg.target.n_layers,
    )
    assert not any(m.type == "PersistentPGDReconLoss" for m in cfg.pd.loss_metrics), (
        "persistent-PGD sources are pinned to a sequence axis in the engine"
        " (init_train_state); positionless deep-linear runs cannot carry the adversary"
    )
    assert_canonical_algorithm_config(cfg)
    build_loss_terms(
        cfg.pd.loss_metrics,
        tuple(sc.name for sc in site_cs),
    )
    target = deep_linear.DeepLinearTargetConfig(
        n_features=cfg.target.n_features,
        n_layers=cfg.target.n_layers,
        sites=site_cs,
        global_batch=cfg.pd.batch_size,
    )
    return BuiltRun(
        pd=cfg.pd,
        runtime=cfg.runtime,
        cadence=cfg.cadence,
        run=run_instance(cfg, run_id),
        target=target,
        data=None,
        ci_fn=ci_arch(cfg.pd.ci_config, resolve_chunkwise=None),
        eval=None,
    )


def run_deep_linear_decomposition(built: BuiltRun, raw_cfg: dict[str, Any], mesh: Mesh) -> None:
    """Build the frozen identity target, then decompose it through the generic engine."""
    target_cfg = built.target
    assert isinstance(target_cfg, deep_linear.DeepLinearTargetConfig)

    dl_cfg = deep_linear.DeepLinearConfig(
        n_features=target_cfg.n_features, n_layers=target_cfg.n_layers
    )
    target = deep_linear.init_deep_linear_target(dl_cfg)
    lm = deep_linear.replicate_target(
        deep_linear.deep_linear_decomposed_model(dl_cfg, target, target_cfg.sites), mesh
    )

    data_key = random.fold_in(random.PRNGKey(built.pd.seed), 17)

    @jax.jit
    def sample_one_hot_batch(step_key: jax.Array) -> jax.Array:
        x = deep_linear.sample_one_hot(step_key, target_cfg.global_batch, target_cfg.n_features)
        return jax.lax.with_sharding_constraint(x, NamedSharding(mesh, P(("replicate", "fsdp"))))

    def sample_batch(step: int) -> jax.Array:
        return sample_one_hot_batch(random.fold_in(data_key, step))

    # `model` is the filter_jit ARG (frozen weights traced, not baked) — closing over an
    # array-bearing eqx model would bake its weights into the HLO.
    @eqx.filter_jit
    def probe_ci(
        model: deep_linear.DeepLinearDecomposedModel, ci_fn: Any
    ) -> tuple[dict[str, jax.Array], dict[str, jax.Array]]:
        probe = deep_linear.one_hot_probe(target_cfg.n_features)
        ci = ci_fn(model.read_activations(probe, ci_fn.input_names), remat=False)
        return ci.lower, ci.upper

    uv_spec = toy_uv_eval.toy_uv_spec(lm, raw_cfg)

    def eval_fn(state: TrainState, now_step: int) -> dict[str, float]:
        ci_lower, ci_upper = probe_ci(lm, state.ci_fn)
        toy_uv_eval.log_uv_figure(
            uv_spec,
            state.components.vu,
            ci_upper,
            now_step,
            wandb_active=built.run.wandb is not None,
        )
        return {
            f"eval/identity_ci_error/{site}": float(identity_ci_error(ci, tolerance=0.1))
            for site, ci in ci_lower.items()
        }

    run_decomposition_training(
        pd=built.pd,
        cadence=built.cadence,
        run=built.run,
        lm=lm,
        ci_fn=built.ci_fn,
        data=built.data,
        remat_recon_forwards=built.runtime.remat_recon_forwards,
        remat_ci_fn=built.runtime.remat_ci_fn,
        ascend_replicate=built.runtime.ascend_replicate,
        compiler_options=built.runtime.compiler_options,
        profile=built.runtime.launch_env.profile,
        sample_batch=sample_batch,
        eval_fn=eval_fn,
        eval_every=built.cadence.train_log_every,
        mesh=mesh,
    )


def main(config: str, group: str | None = None, tags: str | None = None) -> None:
    schema_raw = yaml.safe_load(Path(config).read_text())
    run_id = generate_run_id("param_decomp")
    if group is not None or tags is not None:
        wandb_cfg = dict(schema_raw.get("wandb") or {})
        if group is not None:
            wandb_cfg["group"] = group
        if tags is not None:
            wandb_cfg["tags"] = tags.split(",")
        schema_raw["wandb"] = wandb_cfg
    built = build_deep_linear_built_run(DeepLinearExperimentConfig(**schema_raw), run_id)
    built.run.run_dir.mkdir(parents=True, exist_ok=True)
    setup_logger(built.run.run_dir / "logs.log")
    (built.run.run_dir / LAUNCH_CONFIG_FILENAME).write_text(
        yaml.safe_dump(schema_raw, sort_keys=False)
    )
    mesh = hsdp_mesh()
    run_deep_linear_decomposition(built, schema_raw, mesh)


def cli() -> None:
    fire.Fire(main)


if __name__ == "__main__":
    cli()
