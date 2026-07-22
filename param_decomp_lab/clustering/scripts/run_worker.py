"""Harvest clustering memberships from a JAX single-pool run natively — no torch
component model, no safetensors bridge.

    python -m param_decomp_lab.clustering.scripts.run_worker \
        --run_dir runs/p-761bc061 --n_tokens 50000 --batch_size 16 --n_tokens_per_seq 8

The run is opened with `param_decomp_lab.experiments.lm.load_run.open_jax_run` (the reusable JAX
"open a run for consumption" pattern); the lower-leaky CI from its frozen forward is
sampled per token position and streamed — as a numpy-array dict — into the
`MembershipBuilder`, producing the `ProcessedMemberships` snapshot `pd-cluster-merge`
consumes unchanged.

The JAX forward runs in jax (CPU or one GPU); the `MembershipBuilder` accumulator is
numpy. Pre-tokenized parquet is read with the trainer's own `ShardServer` (never streamed
from HF).
"""

import argparse
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from param_decomp.built_run import DataConfig
from param_decomp.data import BatchSchedule, ShardServer, scan_shards
from param_decomp.log import logger
from param_decomp_lab.clustering.harvest_config import HarvestConfig
from param_decomp_lab.clustering.memberships import MembershipBuilder, flatten_lm_activations
from param_decomp_lab.clustering.paths import clustering_harvest_dir, new_harvest_id
from param_decomp_lab.experiments.lm.load_run import LoadedJaxRun, open_jax_run


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

    data = run.config.data
    assert isinstance(data, DataConfig), f"JAX clustering is LM-only, got {type(data).__name__}"
    schedule = BatchSchedule(scan_shards(data.dir), config.batch_size, config.dataset_seed)
    server = ShardServer(schedule, data.seq_len, process_index=0, process_count=1)

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
        fwd = run.forward(jnp.asarray(tokens))
        sampled = sampled_ci_from_forward(
            fwd.lower_leaky_ci,
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


def run_harvest_to_dir(config: HarvestConfig, harvest_id: str, step: int | None) -> Path:
    """Open the run referenced by `config.model_path`, harvest into the canonical harvest
    dir for `harvest_id`, and return that dir. Used by the standalone CLI and the ensemble
    pipeline (which pre-assigns the harvest id so its dependent merge can find the snapshot).
    """
    run = open_jax_run(Path(config.model_path), step)
    output_dir = clustering_harvest_dir(harvest_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    config.to_file(output_dir / "harvest_config.json")
    logger.info(f"JAX clustering harvest: run {run.run_id} step {run.step}, harvest {harvest_id}")
    harvest_jax_run(run, config, output_dir)
    return output_dir


def get_command(config_path: Path, harvest_id: str, dataset_seed: int) -> str:
    """Shell command for one ensemble member's harvest (config-file driven, seed-overridden)."""
    import shlex

    return shlex.join(
        [
            "python",
            "-m",
            "param_decomp_lab.clustering.scripts.run_worker",
            "--config",
            config_path.as_posix(),
            "--harvest_id",
            harvest_id,
            "--dataset_seed",
            str(dataset_seed),
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
    run_harvest_to_dir(config, harvest_id=harvest_id, step=args.step)


if __name__ == "__main__":
    main()
