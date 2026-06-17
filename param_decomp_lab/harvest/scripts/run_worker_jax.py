"""Harvest a JAX single-pool run natively — no torch component model, no `jsp-export`
safetensors bridge.

    python -m param_decomp_lab.harvest.scripts.run_worker_jax \
        --run_dir runs/p-761bc061 --n_batches 50 --batch_size 16

The run is opened with `jax_single_pool.load_run.open_jax_run` (the reusable JAX
"open a run for consumption" pattern); the frozen forward-only pass it exposes is
turned into the SAME `HarvestBatch` the harvest accumulator consumes, fed to the SAME
`Harvester`, and written via the SAME `HarvestRepo.save_results`. Downstream
autointerp / clustering / app therefore read the output unchanged.

The forward runs in jax (CPU or one GPU); the accumulator is NumPy. The whole harvest
stack is torch-free. Pre-tokenized parquet is read with the trainer's own `ShardServer`
(never streamed from HF).

Sharding mirrors the SLURM launcher's array model: each array task is one rank and
serves its `process_index=rank` slice of every global batch (`ShardServer`), saves its
partial `Harvester` to `worker_states/worker_<rank>.npz`, and the dependent merge job
combines them. A single-process run (no `--rank`) writes the final results directly.
"""

import argparse
from datetime import datetime
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from jax_single_pool.config import DataConfig
from jax_single_pool.data import BatchSchedule, ShardServer, scan_shards
from jax_single_pool.load_run import HarvestForward, LoadedJaxRun, open_jax_run

from param_decomp.log import logger
from param_decomp_lab.harvest.accumulator import Harvester
from param_decomp_lab.harvest.config import HarvestConfig, ParamDecompHarvestConfig
from param_decomp_lab.harvest.repo import HarvestRepo
from param_decomp_lab.harvest.schemas import HarvestBatch, get_harvest_subrun_dir


def harvest_batch_from_forward(
    tokens: np.ndarray, fwd: HarvestForward, activation_threshold: float
) -> HarvestBatch:
    """JAX forward outputs -> the NumPy `HarvestBatch` (lower-leaky CI as
    `causal_importance`, ‖U‖·(x@V) as `component_activation`, firing = CI > threshold)."""
    ci = {site: np.asarray(v) for site, v in fwd.lower_leaky_ci.items()}
    acts = {site: np.asarray(v) for site, v in fwd.component_acts.items()}
    return HarvestBatch(
        tokens=np.asarray(tokens).astype(np.int64),
        firings={site: ci[site] > activation_threshold for site in ci},
        activations={
            site: {"causal_importance": ci[site], "component_activation": acts[site]} for site in ci
        },
        output_probs=np.asarray(fwd.output_probs),
    )


def harvest_jax_run(
    run: LoadedJaxRun,
    config: HarvestConfig,
    output_dir: Path,
    rank_world_size: tuple[int, int] | None,
) -> None:
    method_config = config.method_config
    assert isinstance(method_config, ParamDecompHarvestConfig), (
        "JAX harvest path requires a ParamDecompHarvestConfig"
    )
    activation_threshold = method_config.activation_threshold
    data, seed = run.config.data, run.config.seed
    assert isinstance(data, DataConfig), f"JAX harvest is LM-only, got {type(data).__name__}"
    rank, world_size = rank_world_size if rank_world_size is not None else (0, 1)
    schedule = BatchSchedule(scan_shards(data.dir), config.batch_size, seed)
    server = ShardServer(schedule, data.seq_len, process_index=rank, process_count=world_size)

    harvester = Harvester(
        layers=run.layer_activation_sizes,
        vocab_size=run.vocab_size,
        max_examples_per_component=config.activation_examples_per_component,
        context_tokens_per_side=config.activation_context_tokens_per_side,
        max_examples_per_batch_per_component=config.max_examples_per_batch_per_component,
        collect_component_cooccurrence=config.collect_component_cooccurrence,
    )

    assert isinstance(config.n_batches, int), "JAX harvest needs an explicit n_batches"
    for batch_idx in range(config.n_batches):
        tokens = server.local_batch(batch_idx)
        fwd = run.forward(jnp.asarray(tokens))
        hb = harvest_batch_from_forward(tokens, fwd, activation_threshold)
        harvester.process_batch(hb.tokens, hb.firings, hb.activations, hb.output_probs)
        if rank_world_size is None:
            logger.info(f"{batch_idx + 1}/{config.n_batches} batches")

    logger.info(
        f"Harvest complete: {config.n_batches} batches, {harvester.total_tokens_processed:,} tokens"
    )

    if rank_world_size is None:
        HarvestRepo.save_results(harvester, config, output_dir)
        logger.info(f"Saved results to {output_dir}")
    else:
        state_dir = output_dir / "worker_states"
        state_dir.mkdir(parents=True, exist_ok=True)
        state_path = state_dir / f"worker_{rank}.npz"
        harvester.save(state_path)
        logger.info(f"[Worker {rank}] Saved state to {state_path}")


def get_command(
    config: HarvestConfig, run_dir: Path, rank: int, world_size: int, subrun_id: str
) -> str:
    return (
        f"python -m param_decomp_lab.harvest.scripts.run_worker_jax "
        f"--run_dir {run_dir} "
        f"--n_batches {config.n_batches} "
        f"--batch_size {config.batch_size} "
        f"--activation_threshold {_activation_threshold(config)} "
        f"--rank {rank} "
        f"--world_size {world_size} "
        f"--subrun_id {subrun_id} "
        f"{'--no_cooccurrence' if not config.collect_component_cooccurrence else ''}"
    )


def _activation_threshold(config: HarvestConfig) -> float:
    method_config = config.method_config
    assert isinstance(method_config, ParamDecompHarvestConfig), (
        "JAX harvest path requires a ParamDecompHarvestConfig"
    )
    return method_config.activation_threshold


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run_dir", type=Path, required=True)
    ap.add_argument("--step", type=int, default=None, help="checkpoint step (default: latest)")
    ap.add_argument("--n_batches", type=int, required=True)
    ap.add_argument(
        "--batch_size", type=int, default=HarvestConfig.model_fields["batch_size"].default
    )
    ap.add_argument(
        "--activation_threshold",
        type=float,
        default=ParamDecompHarvestConfig.model_fields["activation_threshold"].default,
    )
    ap.add_argument("--rank", type=int, default=None)
    ap.add_argument("--world_size", type=int, default=None)
    ap.add_argument("--subrun_id", type=str, default=None)
    ap.add_argument(
        "--no_cooccurrence",
        action="store_true",
        help="skip the O(C²) component-cooccurrence matrix (slow on CPU; on for production)",
    )
    args = ap.parse_args()
    assert (args.rank is None) == (args.world_size is None)

    run = open_jax_run(args.run_dir, args.step)
    subrun_id = args.subrun_id or "h-" + datetime.now().strftime("%Y%m%d_%H%M%S")
    config = HarvestConfig(
        method_config=ParamDecompHarvestConfig(
            wandb_path=run.run_id, activation_threshold=args.activation_threshold
        ),
        n_batches=args.n_batches,
        batch_size=args.batch_size,
        collect_component_cooccurrence=not args.no_cooccurrence,
    )
    output_dir = get_harvest_subrun_dir(run.run_id, subrun_id)
    rank_world_size = (args.rank, args.world_size) if args.rank is not None else None
    if rank_world_size is not None:
        logger.info(
            f"Distributed JAX harvest: rank {args.rank}/{args.world_size}, subrun {subrun_id}"
        )
    else:
        logger.info(f"JAX harvest: run {run.run_id} step {run.step}, subrun {subrun_id}")
    harvest_jax_run(run, config, output_dir, rank_world_size)


if __name__ == "__main__":
    main()
