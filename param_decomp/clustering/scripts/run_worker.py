"""Build clustering memberships from one finished JAX VPD run.

The worker restores the run with `open_jax_run`, evaluates lower-leaky CI for the selected
corpus positions, and streams NumPy batches into `MembershipBuilder`. It writes the same
membership files consumed by the existing clustering merge step."""

import argparse
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from param_decomp.clustering.harvest_config import HarvestConfig
from param_decomp.clustering.memberships import MembershipBuilder, flatten_lm_activations
from param_decomp.clustering.paths import clustering_harvest_dir, new_harvest_id
from param_decomp.core.ci_fn import PlacedCIFn, evaluate_ci
from param_decomp.core.log import logger
from param_decomp.core.model import PlacedModel
from param_decomp.experiments.lm.load_run import LoadedJaxRun, open_jax_run
from param_decomp.infra.dataset_store import read_dataset_meta
from param_decomp.pretrain.batch_data import BatchSchedule, ShardServer, scan_shards


@eqx.filter_jit
def _lower_leaky_ci_forward(
    placed: PlacedModel, ci_fn: PlacedCIFn, token_ids: jax.Array
) -> dict[str, jax.Array]:
    clean = placed.model.clean_forward(
        token_ids, capture_keys=ci_fn.fn.capture_keys, placement=placed.placement
    )
    ci = evaluate_ci(ci_fn, clean.captures, remat=False)
    return {site: ci.lower[site].astype(jnp.float32) for site in placed.site_names}


def lower_leaky_ci(run: LoadedJaxRun, token_ids: jax.Array) -> dict[str, jax.Array]:
    """The clustering-only CI forward over one restored run."""
    with jax.set_mesh(run.mesh):
        return _lower_leaky_ci_forward(run.placed, run.ci_fn, token_ids)


def sampled_ci_from_forward(
    lower_leaky_ci: dict[str, jnp.ndarray],
    *,
    n_tokens_per_seq: int | None,
    use_all_tokens_per_seq: bool,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Per-site lower-leaky CI `(B, T, C)` -> `(samples, C)`, sampling token positions
    via `flatten_lm_activations`."""
    site_ci = {site: np.asarray(ci) for site, ci in lower_leaky_ci.items()}
    batch_size, n_ctx, _ = next(iter(site_ci.values())).shape
    return {
        site: flatten_lm_activations(
            ci,
            batch_size=batch_size,
            n_ctx=n_ctx,
            n_tokens_per_seq=n_tokens_per_seq,
            use_all_tokens_per_seq=use_all_tokens_per_seq,
            rng=rng,
        )
        for site, ci in site_ci.items()
    }


def harvest_jax_run(run: LoadedJaxRun, config: HarvestConfig, output_dir: Path) -> None:
    assert config.use_all_tokens_per_seq or config.n_tokens_per_seq is not None, (
        "n_tokens_per_seq required when use_all_tokens_per_seq is False"
    )

    data = run.deliverable.data
    schedule = BatchSchedule(scan_shards(data.dir), config.batch_size, config.dataset_seed)
    server = ShardServer(
        schedule, read_dataset_meta(data.dir).seq_len, process_index=0, process_count=1
    )

    rng = np.random.default_rng(config.dataset_seed)
    builder = MembershipBuilder(
        activation_threshold=config.activation_threshold,
        filter_dead_threshold=config.filter_dead_threshold,
        filter_dead_stat=config.filter_dead_stat,
        filter_modules=config.filter_modules,
    )

    n_collected = 0
    batch_idx = 0
    while n_collected < config.n_tokens:
        tokens = server.local_batch(batch_idx)
        batch_size, n_ctx = tokens.shape
        sampled = sampled_ci_from_forward(
            lower_leaky_ci(run, jnp.asarray(tokens)),
            n_tokens_per_seq=config.n_tokens_per_seq,
            use_all_tokens_per_seq=config.use_all_tokens_per_seq,
            rng=rng,
        )

        tokens_per_seq = n_ctx if config.use_all_tokens_per_seq else config.n_tokens_per_seq
        assert tokens_per_seq is not None
        batch_take = min(batch_size * tokens_per_seq, config.n_tokens - n_collected)
        builder.add_batch({site: ci[:batch_take] for site, ci in sampled.items()})

        n_collected += batch_take
        batch_idx += 1
        logger.info(f"{n_collected}/{config.n_tokens} tokens ({batch_idx} batches)")

    logger.info(f"Collected {n_collected} token activations (requested {config.n_tokens})")
    processed = builder.finalize()
    logger.info(f"Saving: {processed.n_components_alive} alive, {processed.n_samples} samples")
    processed.save(output_dir)
    logger.info(f"Harvest complete: {output_dir}")


def run_harvest_to_dir(
    config: HarvestConfig, harvest_id: str, step: int | None, data_root: Path
) -> Path:
    """Open the run referenced by `config.model_path`, harvest into the canonical harvest
    dir for `harvest_id`, and return that dir. Used by the standalone CLI and the ensemble
    pipeline (which pre-assigns the harvest id so its dependent merge can find the snapshot).
    """
    run = open_jax_run(Path(config.model_path), step, data_root=data_root)
    output_dir = clustering_harvest_dir(data_root, harvest_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    config.to_file(output_dir / "harvest_config.json")
    logger.info(f"JAX clustering harvest: run {run.run_id} step {run.step}, harvest {harvest_id}")
    harvest_jax_run(run, config, output_dir)
    return output_dir


def get_command(config_path: Path, harvest_id: str, dataset_seed: int, data_root: Path) -> str:
    """Shell command for one ensemble member's harvest (config-file driven, seed-overridden)."""
    import shlex

    return shlex.join(
        [
            "python",
            "-m",
            "param_decomp.clustering.scripts.run_worker",
            "--config",
            config_path.as_posix(),
            "--harvest_id",
            harvest_id,
            "--dataset_seed",
            str(dataset_seed),
            "--data_root",
            data_root.as_posix(),
        ]
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--config",
        type=Path,
        default=None,
        help="HarvestConfig JSON. If given, --run_dir/--n_tokens/etc. are ignored.",
    )
    ap.add_argument("--harvest_id", type=str, default=None, help="pre-assigned harvest id")
    ap.add_argument("--data_root", type=Path, required=True)
    ap.add_argument("--step", type=int, default=None, help="checkpoint step (default: latest)")
    ap.add_argument("--run_dir", type=Path, default=None)
    ap.add_argument("--n_tokens", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--n_tokens_per_seq", type=int, default=None)
    ap.add_argument("--use_all_tokens_per_seq", action="store_true")
    ap.add_argument("--dataset_seed", type=int, default=0)
    ap.add_argument("--activation_threshold", type=float, default=0.0)
    ap.add_argument("--filter_dead_threshold", type=float, default=0.001)
    args = ap.parse_args()

    if args.config is not None:
        config = HarvestConfig.from_file(args.config)
        config = config.model_copy(update={"dataset_seed": args.dataset_seed})
    else:
        assert args.run_dir is not None and args.n_tokens is not None, (
            "--run_dir and --n_tokens are required when --config is not given"
        )
        config = HarvestConfig(
            model_path=args.run_dir,
            batch_size=args.batch_size,
            n_tokens=args.n_tokens,
            n_tokens_per_seq=args.n_tokens_per_seq,
            use_all_tokens_per_seq=args.use_all_tokens_per_seq,
            dataset_seed=args.dataset_seed,
            activation_threshold=args.activation_threshold,
            filter_dead_threshold=args.filter_dead_threshold,
        )

    harvest_id = args.harvest_id or new_harvest_id()
    run_harvest_to_dir(config, harvest_id=harvest_id, step=args.step, data_root=args.data_root)


if __name__ == "__main__":
    main()
