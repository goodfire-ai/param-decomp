"""Harvest clustering memberships from a JAX single-pool run natively — no torch
component model, no `jsp-export` safetensors bridge.

    python -m param_decomp_lab.clustering.scripts.run_worker_jax \
        --run_dir runs/p-761bc061 --n_tokens 50000 --batch_size 16 --n_tokens_per_seq 8

The run is opened with `jax_single_pool.load_run.open_jax_run` (the reusable JAX
"open a run for consumption" pattern); the lower-leaky CI from its frozen forward is
sampled per token position and streamed — as the SAME torch-tensor dict the torch
`collect_memberships` builds — into the SAME `MembershipBuilder`, producing the SAME
`ProcessedMemberships` snapshot `pd-cluster-merge` consumes unchanged.

The JAX forward runs in jax (CPU or one GPU); the `MembershipBuilder` accumulator stays
torch. This worker imports both — the only place the two stacks meet. Pre-tokenized
parquet is read with the trainer's own `ShardServer` (never streamed from HF).
"""

import argparse
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import torch
from jax_single_pool.data import BatchSchedule, ShardServer, scan_shards
from jax_single_pool.load_run import LoadedJaxRun, open_jax_run

from param_decomp.log import logger
from param_decomp_lab.clustering.harvest_config import HarvestConfig
from param_decomp_lab.clustering.memberships import MembershipBuilder, flatten_lm_activations
from param_decomp_lab.clustering.paths import clustering_harvest_dir, new_harvest_id


def _to_torch(array: object) -> torch.Tensor:
    """Host a JAX/numpy array as a CPU torch tensor."""
    return torch.from_numpy(np.array(np.asarray(array)))


def sampled_ci_from_forward(
    lower_leaky_ci: dict[str, jnp.ndarray],
    *,
    n_tokens_per_seq: int | None,
    use_all_tokens_per_seq: bool,
    rng: torch.Generator,
) -> dict[str, torch.Tensor]:
    """Per-site lower-leaky CI `(B, T, C)` -> `(samples, C)`, sampling token positions
    the same way the torch `collect_memberships` path does (`flatten_lm_activations`)."""
    site_ci = {site: _to_torch(ci) for site, ci in lower_leaky_ci.items()}
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
    schedule = BatchSchedule(scan_shards(data.dir), config.batch_size, config.dataset_seed)
    server = ShardServer(schedule, data.seq_len, process_index=0, process_count=1)

    rng = torch.Generator().manual_seed(config.dataset_seed)
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run_dir", type=Path, required=True)
    ap.add_argument("--step", type=int, default=None, help="checkpoint step (default: latest)")
    ap.add_argument("--n_tokens", type=int, required=True)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--n_tokens_per_seq", type=int, default=None)
    ap.add_argument("--use_all_tokens_per_seq", action="store_true")
    ap.add_argument("--dataset_seed", type=int, default=0)
    ap.add_argument("--activation_threshold", type=float, default=0.0)
    ap.add_argument("--filter_dead_threshold", type=float, default=0.001)
    args = ap.parse_args()

    run = open_jax_run(args.run_dir, args.step)
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
    harvest_id = new_harvest_id()
    output_dir = clustering_harvest_dir(harvest_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    config.to_file(output_dir / "harvest_config.json")
    logger.info(f"JAX clustering harvest: run {run.run_id} step {run.step}, harvest {harvest_id}")
    harvest_jax_run(run, config, output_dir)


if __name__ == "__main__":
    main()
