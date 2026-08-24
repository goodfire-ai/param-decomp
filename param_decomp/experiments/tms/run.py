"""Run a TMS (Toy Model of Superposition) parameter decomposition on CPU
(`python -m param_decomp.experiments.tms.run <config.yaml>`).

The toy domains live lab-side and call the generic core engine
(`param_decomp.core.run.run_decomposition_training`) as a library — the core itself carries
zero toy-specific code. A TMS run pretrains its tiny target from scratch in-process (the
Anthropic `mean((|x|-out)^2)` objective), then decomposes it through the same engine the
LM uses, validating via the ground-truth identity-CI metric logged every train-log step.

These toys train in seconds; the runner is synchronous, on CPU, in the main venv (no SLURM
or CUDA). It mints its own `p-<8hex>` run id (toys do not go through the LM submit path);
pass `--run-id` to resume an existing run from its checkpoints.
"""

from pathlib import Path
from typing import Literal

import equinox as eqx
import fire
import jax
import numpy as np
import yaml
from jax import random
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from param_decomp.core import placement
from param_decomp.core.built_run import BuiltRun
from param_decomp.core.ci_fn import CIFn
from param_decomp.core.components import SiteC, nonlinearity_partitions
from param_decomp.core.configs import Checkpointing
from param_decomp.core.eval_schedule import Every
from param_decomp.core.log import setup_logger
from param_decomp.core.metrics import LogRecord, MetricValue
from param_decomp.core.model import BATCH_AXES, PlacedModel, Positionless
from param_decomp.core.nonlinearity_eval import (
    make_nonlinearity_eval_step,
    nonlinearity_log_entries,
)
from param_decomp.core.objective import build_objective
from param_decomp.core.run import (
    EvalInvocation,
    EvalOperation,
    Evaluation,
    MetricsSink,
    install_sigterm_flag,
    run_decomposition_training,
)
from param_decomp.core.sharding import single_device_mesh
from param_decomp.experiments import toy_uv_eval
from param_decomp.experiments.config import (
    apply_wandb_cli_overrides,
    pin_launch_config,
    run_instance,
)
from param_decomp.experiments.eval_config import EvalConfig
from param_decomp.experiments.tms.config import TMSExperimentConfig
from param_decomp.experiments.toy_config import build_toy_ci_arch
from param_decomp.experiments.toy_eval import ToyRun, make_toy_evaluation_operations
from param_decomp.infra.run_files import generate_run_id
from param_decomp.targets import tms

TMSRun = ToyRun[tms.TMSTargetConfig]


def build_tms_built_run(cfg: TMSExperimentConfig, run_id: str, data_root: Path) -> TMSRun:
    """Convert the canonical TMS schema to the engine's `BuiltRun` bundle via the shared
    helpers. TMS keeps its typed toy eval plan: fresh PGD runs at the configured cadence, while
    `UVPlots` and the native identity-CI diagnostics use the single-feature probe."""
    site_cs = tms.canonical_site_cs(
        tuple(SiteC(s.name, s.C) for s in cfg.decomposition.sites.sites)
    )
    build_objective(
        cfg.pd.loss_metrics,
        tuple(sc.name for sc in site_cs),
    )
    tms_cfg = tms.TMSConfig(
        n_features=cfg.target.n_features,
        n_hidden=cfg.target.n_hidden,
        n_hidden_layers=cfg.target.n_hidden_layers,
        hidden_layer_init=cfg.target.hidden_layer_init,
        init_bias_to_zero=cfg.target.init_bias_to_zero,
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
        cadence=cfg.cadence,
        run=run_instance(cfg, run_id, data_root, None),
        target=target,
        data=None,
        ci_fn=build_toy_ci_arch(
            cfg.decomposition.ci,
            tms.site_input_tap_keys(tuple(sc.name for sc in site_cs)),
            tms.site_specs(tms_cfg, site_cs),
        ),
    )


def pretrained_tms_model(
    target_cfg: tms.TMSTargetConfig, mesh: Mesh, is_main: bool
) -> tms.TMSDecomposedModel:
    """Build + from-scratch-pretrain the frozen TMS target and wrap it as the decomposed
    model — one `eqx.Module` carrying the TMS weights as a field and the decomposition
    contract as methods."""
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
    return tms.replicate_target(
        tms.tms_decomposed_model(tms_cfg, target, tms.site_specs(tms_cfg, target_cfg.sites)), mesh
    )


def sparse_feature_sampler(
    mesh: Mesh,
    batch_size: int,
    n_features: int,
    feature_probability: float,
    generation_type: tms.TMSGenerationType,
):
    """A jitted `sample(step_key) -> [batch, n_features]` batch-sharded sparse-feature
    draw; the caller owns the per-step key derivation."""

    @jax.jit
    def sample(step_key: jax.Array) -> jax.Array:
        x = tms.sample_sparse_features(
            step_key, batch_size, n_features, feature_probability, generation_type
        )
        return jax.sharding.reshard(x, NamedSharding(mesh, P(BATCH_AXES)))

    return sample


# `model` is the filter_jit ARG (frozen TMS weights traced, not baked) — closing over an
# array-bearing eqx model would bake its weights into the HLO; `n_features` is static.
@eqx.filter_jit
def single_feature_ci(
    model: tms.TMSDecomposedModel, ci_fn: CIFn, n_features: int
) -> tuple[dict[str, jax.Array], dict[str, jax.Array]]:
    probe = tms.single_feature_probe(n_features)
    ci = ci_fn(
        model.clean_forward(probe, ci_fn.capture_keys, placement=None).captures,
        remat=False,
        placement=None,
    )
    return ci.lower, ci.upper


def tms_ground_truth_operation(
    model: tms.TMSDecomposedModel,
    n_features: int,
    total_steps: int,
    checkpointing: Checkpointing,
    train_log_every: int,
) -> EvalOperation[EvalInvocation]:
    """The TMS-native ground-truth pass: the `lower_leaky` CI of the single-feature probe
    scored as per-site `IdentityCIError` every train-log step, plus the per-site-permuted
    CI heatmap image alongside each checkpoint."""

    # The frozen `hidden_layers.*` sites (the `-id` variant) target DENSE recovery — every
    # direction stays live — not identity; canonical torch parity (`tms_40-10-id_config.yaml`
    # `dense_patterns: [hidden_layers.0]`). `linear1`/`linear2` target identity.
    ci_permutation: dict[str, Literal["identity", "dense"]] = {
        site: "dense" if site.startswith("hidden_layers.") else "identity"
        for site in model.site_names
    }

    partitions = nonlinearity_partitions(model.sites)
    nonlinearity_eval_step = make_nonlinearity_eval_step(partitions, {})

    def ground_truth_eval(context: EvalInvocation) -> LogRecord:
        state, now_step = context.state, context.now_step
        ci_lower, ci_upper = single_feature_ci(model, state.decomposition.ci_fn, n_features)
        figures: LogRecord = {}
        if toy_uv_eval.permuted_ci_heatmap_due(now_step, total_steps, checkpointing):
            figures = toy_uv_eval.render_permuted_ci_heatmap(
                ci_lower,
                ci_upper,
                ci_permutation,
            )
        metrics: dict[str, MetricValue] = dict(figures)
        metrics.update(
            {
                f"eval/identity_ci_error/{site}": float(tms.identity_ci_error(ci, tolerance=0.1))
                for site, ci in ci_lower.items()
                if ci_permutation[site] == "identity"
            }
        )
        ci_means = {name: np.asarray(value).mean(0) for name, value in ci_lower.items()}
        metrics.update(
            nonlinearity_log_entries(
                nonlinearity_eval_step(state.decomposition.components), ci_means, partitions
            )
        )
        return metrics

    return EvalOperation(schedule=Every(train_log_every), run=ground_truth_eval)


def run_tms_decomposition(built: TMSRun, eval_config: EvalConfig | None, mesh: Mesh) -> None:
    """Build + pretrain the TMS target, then decompose it through the generic engine.

    The batch entering the decomposed model IS the raw input `x`."""
    target_cfg = built.target
    is_main = jax.process_index() == 0
    model = pretrained_tms_model(target_cfg, mesh, is_main)
    placed_model = PlacedModel(
        model=model, placement=placement.from_config("ddp", mesh, model.sites)
    )

    data_key = random.fold_in(random.PRNGKey(built.pd.seed), 17)

    def make_sampler(batch_size: int):
        return sparse_feature_sampler(
            mesh,
            batch_size,
            target_cfg.n_features,
            target_cfg.feature_probability,
            target_cfg.data_generation_type,
        )

    sample_train = make_sampler(target_cfg.global_batch)

    def sample_batch(step: int) -> jax.Array:
        return sample_train(random.fold_in(data_key, step))

    operations = [
        tms_ground_truth_operation(
            model,
            target_cfg.n_features,
            built.pd.steps,
            built.cadence.checkpointing,
            built.cadence.train_log_every,
        )
    ]
    if eval_config is not None:
        eval_sampler = make_sampler(eval_config.batch_size)
        operations.extend(
            make_toy_evaluation_operations(
                eval_config,
                built.pd.seed,
                compiler_options={},
                model=placed_model,
                ci_capture_keys=built.ci_fn.capture_keys,
                mesh=mesh,
                sample_eval_batch=lambda index: eval_sampler(
                    random.fold_in(data_key, built.pd.steps + index)
                ),
                probe_ci=lambda state: single_feature_ci(
                    model, state.decomposition.ci_fn, target_cfg.n_features
                )[1],
                wandb_configured=built.run.wandb is not None,
            )
        )
    evaluation = Evaluation(tuple(operations), lambda invocation: invocation)

    sink = MetricsSink.for_run(built.run, jax.process_index() == 0)
    run_decomposition_training(
        pd=built.pd,
        cadence=built.cadence,
        run=built.run,
        model=placed_model,
        ci_fn=built.ci_fn,
        positions=Positionless(),
        # A toy trains in seconds on one CPU device: nothing to trade memory for, and no
        # GPU collectives for an XLA flag to tune.
        remat_recon_forwards=False,
        remat_ci_fn=False,
        compiler_options={},
        sample_batch=sample_batch,
        evaluation=evaluation,
        sink=sink,
        profiling=None,
    )


def main(
    config: str,
    data_root: Path,
    run_id: str | None = None,
    group: str | None = None,
    tags: str | tuple[str, ...] | None = None,
) -> None:
    schema_raw = yaml.safe_load(Path(config).read_text())
    data_root = Path(data_root)
    if run_id is None:
        # Fresh invocation: mint a new identity; resume an existing run's checkpoints by
        # passing --run-id, exactly as the LM trainer does.
        run_id = generate_run_id("param_decomp")
    apply_wandb_cli_overrides(schema_raw, group, tags)
    cfg = TMSExperimentConfig(**schema_raw)
    built = build_tms_built_run(cfg, run_id, data_root)
    built.run.run_dir.mkdir(parents=True, exist_ok=True)
    setup_logger(built.run.run_dir / "logs.log")
    pin_launch_config(built.run.run_dir, yaml.safe_dump(schema_raw, sort_keys=False))
    install_sigterm_flag()
    mesh = single_device_mesh()
    run_tms_decomposition(built, cfg.eval, mesh)


def cli() -> None:
    fire.Fire(main)


if __name__ == "__main__":
    cli()
