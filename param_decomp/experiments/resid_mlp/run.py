"""Run a ResidualMLP parameter decomposition on CPU
(`python -m param_decomp.experiments.resid_mlp.run <config.yaml>`).

The SPD/APD residual-stream toy lives lab-side and calls the generic core engine
(`param_decomp.core.run.run_decomposition_training`) as a library. The target pretrains from
scratch in-process (the `act_fn(coeffs·x) + x` read-off objective), then decomposes through
the same engine the LM uses, validating via the ground-truth identity-CI metric.

These toys train in seconds; the runner is synchronous, on CPU, in the main venv
(no SLURM / `param_decomp.core.run` / CUDA). It mints its own `p-<8hex>` run id;
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
from param_decomp.core.slow_eval import dense_ci_error
from param_decomp.experiments import toy_uv_eval
from param_decomp.experiments.config import (
    pin_launch_config,
    run_instance,
)
from param_decomp.experiments.eval_config import EvalConfig
from param_decomp.experiments.resid_mlp.config import ResidMLPExperimentConfig
from param_decomp.experiments.toy_config import build_toy_ci_arch
from param_decomp.experiments.toy_eval import ToyRun, make_toy_evaluation_operations
from param_decomp.infra.run_files import generate_run_id
from param_decomp.targets import resid_mlp

ResidMLPRun = ToyRun[resid_mlp.ResidMLPTargetConfig]


def build_resid_mlp_built_run(
    cfg: ResidMLPExperimentConfig, run_id: str, data_root: Path
) -> ResidMLPRun:
    """Convert the canonical ResidMLP schema to the engine's `BuiltRun` bundle via the shared
    helpers. ResidMLP keeps its typed toy eval plan: fresh PGD runs at the configured cadence, while
    `UVPlots` and the native identity-CI diagnostics use the single-feature probe."""
    site_cs = resid_mlp.canonical_site_cs(
        tuple(SiteC(s.name, s.C) for s in cfg.decomposition.sites.sites)
    )
    build_objective(
        cfg.pd.loss_metrics,
        tuple(sc.name for sc in site_cs),
    )
    resid_cfg = resid_mlp.ResidMLPConfig(
        n_features=cfg.target.n_features,
        d_embed=cfg.target.d_embed,
        d_mlp=cfg.target.d_mlp,
        n_layers=cfg.target.n_layers,
        act_fn_name=cfg.target.act_fn_name,
        in_bias=cfg.target.in_bias,
        out_bias=cfg.target.out_bias,
        fixed_identity_embedding=cfg.target.fixed_identity_embedding,
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
        pretrain_label_type=cfg.target.pretrain.label_type,
        pretrain_loss_type=cfg.target.pretrain.loss_type,
        pretrain_use_trivial_label_coeffs=cfg.target.pretrain.use_trivial_label_coeffs,
        pretrain_importance_val=cfg.target.pretrain.importance_val,
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
            tuple(sc.name for sc in site_cs),
            resid_mlp.site_specs(resid_cfg, site_cs),
        ),
    )


def run_resid_mlp_decomposition(
    built: ResidMLPRun, eval_config: EvalConfig | None, mesh: Mesh
) -> None:
    """Build + pretrain the ResidMLP target, then decompose it through the generic engine.

    The batch entering the decomposed model is `x @ W_E` (`W_E` is carried inside the
    frozen target, not decomposed). Its native eval operation reads the `lower_leaky` CI of
    the single-feature probe (embedded through `W_E`) and logs the ground-truth
    `IdentityCIError` per site every train-log step, plus the
    per-site-permuted CI heatmap image alongside each checkpoint
    (`toy_uv_eval.render_permuted_ci_heatmap` — `mlp_out` permutes toward dense, not identity)."""
    target_cfg = built.target
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
        target_cfg.pretrain_label_type,
        target_cfg.pretrain_loss_type,
        target_cfg.pretrain_use_trivial_label_coeffs,
        target_cfg.pretrain_importance_val,
    )
    # The model IS the frozen target: one `eqx.Module` carries the ResidMLP weights as a field
    # and the decomposition contract as methods.
    model = resid_mlp.replicate_target(
        resid_mlp.resid_mlp_decomposed_model(
            resid_cfg, target, resid_mlp.site_specs(resid_cfg, target_cfg.sites)
        ),
        mesh,
    )
    placed_model = PlacedModel(
        model=model, placement=placement.from_config("ddp", mesh, model.sites)
    )

    data_key = random.fold_in(random.PRNGKey(built.pd.seed), 17)

    # `tgt` is the filter_jit ARG (frozen `W_E` traced, not baked) — closing over an
    # array-bearing eqx target would bake its weights into the HLO.
    def make_sampler(batch_size: int):
        @eqx.filter_jit
        def sample(tgt: resid_mlp.ResidMLPTarget, step_key: jax.Array) -> jax.Array:
            x = resid_mlp.sample_sparse_features(
                step_key,
                batch_size,
                target_cfg.n_features,
                target_cfg.feature_probability,
                target_cfg.data_generation_type,
            )
            residual = resid_mlp.resid_mlp_input_residual(tgt, x)
            return jax.sharding.reshard(residual, NamedSharding(mesh, P(BATCH_AXES)))

        return sample

    sample_train = make_sampler(target_cfg.global_batch)

    def sample_batch(step: int) -> jax.Array:
        return sample_train(model.target, random.fold_in(data_key, step))

    # `model` is the filter_jit ARG (frozen weights traced, not baked).
    @eqx.filter_jit
    def single_feature_ci(
        model: resid_mlp.ResidMLPDecomposedModel, ci_fn: CIFn
    ) -> tuple[dict[str, jax.Array], dict[str, jax.Array]]:
        resid = resid_mlp.single_feature_probe(target_cfg.n_features) @ model.target.W_E
        ci = ci_fn(
            model.clean_forward(resid, ci_fn.capture_keys, placement=None).captures,
            remat=False,
            placement=None,
        )
        return ci.lower, ci.upper

    # `mlp_out` targets DENSE recovery (every d_mlp direction stays live), not identity —
    # torch parity (`resid_mlp{1,2,3}_config.yaml` `dense_patterns: [layers.*.mlp_out]`).
    # `mlp_in` targets identity.
    ci_permutation: dict[str, Literal["identity", "dense"]] = {
        site: "dense" if site.endswith(resid_mlp.MLP_OUT) else "identity"
        for site in model.site_names
    }

    partitions = nonlinearity_partitions(model.sites)
    nonlinearity_eval_step = make_nonlinearity_eval_step(partitions, {})

    def ground_truth_eval(context: EvalInvocation) -> LogRecord:
        state, now_step = context.state, context.now_step
        ci_lower, ci_upper = single_feature_ci(model, state.decomposition.ci_fn)
        figures: LogRecord = {}
        if toy_uv_eval.permuted_ci_heatmap_due(
            now_step, built.pd.steps, built.cadence.checkpointing
        ):
            figures = toy_uv_eval.render_permuted_ci_heatmap(
                ci_lower,
                ci_upper,
                ci_permutation,
            )
        metrics: dict[str, MetricValue] = dict(figures)
        metrics.update(
            {
                f"eval/identity_ci_error/{site}": float(
                    resid_mlp.identity_ci_error(ci, tolerance=0.1)
                )
                for site, ci in ci_lower.items()
                if ci_permutation[site] == "identity"
            }
        )
        metrics.update(
            {
                f"eval/dense_ci_error/{site}": float(
                    dense_ci_error(np.asarray(ci_lower[site]), k=target_cfg.d_mlp, tolerance=0.1)
                )
                for site in ci_lower
                if ci_permutation[site] == "dense"
            }
        )
        ci_means = {name: np.asarray(value).mean(0) for name, value in ci_lower.items()}
        metrics.update(
            nonlinearity_log_entries(
                nonlinearity_eval_step(state.decomposition.components), ci_means, partitions
            )
        )
        return metrics

    operations = [
        EvalOperation(schedule=Every(built.cadence.train_log_every), run=ground_truth_eval)
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
                    model.target, random.fold_in(data_key, built.pd.steps + index)
                ),
                probe_ci=lambda state: single_feature_ci(model, state.decomposition.ci_fn)[1],
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
    cfg = ResidMLPExperimentConfig(**schema_raw)
    built = build_resid_mlp_built_run(cfg, run_id, data_root)
    built.run.run_dir.mkdir(parents=True, exist_ok=True)
    setup_logger(built.run.run_dir / "logs.log")
    pin_launch_config(built.run.run_dir, yaml.safe_dump(schema_raw, sort_keys=False))
    install_sigterm_flag()
    mesh = single_device_mesh()
    run_resid_mlp_decomposition(built, cfg.eval, mesh)


def cli() -> None:
    fire.Fire(main)


if __name__ == "__main__":
    cli()
