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
import yaml
from jax import random
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from param_decomp.core import placement
from param_decomp.core.built_run import BuiltRun
from param_decomp.core.ci_fn import CIFn
from param_decomp.core.components import SiteC
from param_decomp.core.eval_schedule import Every
from param_decomp.core.log import setup_logger
from param_decomp.core.metrics import LogRecord, MetricValue
from param_decomp.core.model import Positionless
from param_decomp.core.objective import build_objective
from param_decomp.core.run import (
    EvalInvocation,
    EvalOperation,
    Evaluation,
    MetricsSink,
    run_decomposition_training,
)
from param_decomp.core.sharding import single_device_mesh
from param_decomp.experiments import toy_uv_eval
from param_decomp.experiments.config import (
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
        ci_fn=build_toy_ci_arch(cfg.decomposition.ci),
    )


def run_tms_decomposition(built: TMSRun, eval_config: EvalConfig | None, mesh: Mesh) -> None:
    """Build + pretrain the TMS target, then decompose it through the generic engine.

    The batch entering the decomposed model IS the raw input `x`. Its native eval
    operation reads the `lower_leaky` CI of the single-feature probe and logs the
    ground-truth `IdentityCIError` per site every train-log step, plus the
    per-site-permuted CI heatmap image alongside each checkpoint (`toy_uv_eval.render_permuted_ci_heatmap` — the
    frozen `hidden_layers.*` sites permute toward dense, not identity)."""
    target_cfg = built.target
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

    def make_sampler(batch_size: int):
        @jax.jit
        def sample(step_key: jax.Array) -> jax.Array:
            x = tms.sample_sparse_features(
                step_key,
                batch_size,
                target_cfg.n_features,
                target_cfg.feature_probability,
                target_cfg.data_generation_type,
            )
            return jax.lax.with_sharding_constraint(
                x, NamedSharding(mesh, P(("replicate", "fsdp")))
            )

        return sample

    sample_train = make_sampler(target_cfg.global_batch)

    def sample_batch(step: int) -> jax.Array:
        return sample_train(random.fold_in(data_key, step))

    # `model` is the filter_jit ARG (frozen TMS weights traced, not baked) — closing over an
    # array-bearing eqx model would bake its weights into the HLO.
    @eqx.filter_jit
    def single_feature_ci(
        model: tms.TMSDecomposedModel, ci_fn: CIFn
    ) -> tuple[dict[str, jax.Array], dict[str, jax.Array]]:
        probe = tms.single_feature_probe(target_cfg.n_features)
        ci = ci_fn(model.read_activations(probe, ci_fn.input_names), remat=False)
        return ci.lower, ci.upper

    # The frozen `hidden_layers.*` sites (the `-id` variant) target DENSE recovery — every
    # direction stays live — not identity; canonical torch parity (`tms_40-10-id_config.yaml`
    # `dense_patterns: [hidden_layers.0]`). `linear1`/`linear2` target identity.
    ci_permutation: dict[str, Literal["identity", "dense"]] = {
        site: "dense" if site.startswith("hidden_layers.") else "identity"
        for site in model.site_names
    }

    def ground_truth_eval(context: EvalInvocation) -> LogRecord:
        state, now_step = context.state, context.now_step
        ci_lower, ci_upper = single_feature_ci(model, state.decomposition.ci_fn)
        figures: LogRecord = {}
        if toy_uv_eval.permuted_ci_heatmap_due(now_step, built.pd.steps, built.cadence.save_every):
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
                model=model,
                mesh=mesh,
                sample_eval_batch=lambda index: eval_sampler(
                    random.fold_in(data_key, built.pd.steps + index)
                ),
                probe_ci=lambda state: single_feature_ci(model, state.decomposition.ci_fn)[1],
                wandb_configured=built.run.wandb is not None,
            )
        )
    evaluation = Evaluation(
        tuple(operations), lambda state, now_step: EvalInvocation(state, now_step)
    )

    sink = MetricsSink.for_run(built.run, jax.process_index() == 0)
    run_decomposition_training(
        pd=built.pd,
        cadence=built.cadence,
        run=built.run,
        model=model,
        ci_fn=built.ci_fn,
        positions=Positionless(),
        # A toy trains in seconds on one CPU device: nothing to trade memory for, and no
        # GPU collectives for an XLA flag to tune.
        remat_recon_forwards=False,
        remat_ci_fn=False,
        ascend_replicate=False,
        compiler_options={},
        sample_batch=sample_batch,
        evaluation=evaluation,
        sink=sink,
        mesh=mesh,
        placement_rules=placement.from_config("ddp", mesh, model.sites),
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
    cfg = TMSExperimentConfig(**schema_raw)
    built = build_tms_built_run(cfg, run_id, data_root)
    built.run.run_dir.mkdir(parents=True, exist_ok=True)
    setup_logger(built.run.run_dir / "logs.log")
    pin_launch_config(built.run.run_dir, yaml.safe_dump(schema_raw, sort_keys=False))
    mesh = single_device_mesh()
    run_tms_decomposition(built, cfg.eval, mesh)


def cli() -> None:
    fire.Fire(main)


if __name__ == "__main__":
    cli()
