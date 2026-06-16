"""Harvest a JAX single-pool run natively — no torch component model, no `jsp-export`
safetensors bridge.

    python -m param_decomp_lab.harvest.scripts.run_worker_jax \
        --run_dir runs/p-761bc061 --n_batches 50 --batch_size 16

The run is opened with `jax_single_pool.load_run.open_jax_run` (the reusable JAX
"open a run for consumption" pattern); the frozen forward-only pass it exposes is
turned into the SAME `HarvestBatch` the torch harvest fn produces, fed to the SAME
`Harvester`, and written via the SAME `HarvestRepo.save_results`. Downstream
autointerp / clustering / app therefore read the output unchanged.

The JAX forward runs in jax (CPU or one GPU); the accumulator stays torch. This worker
imports both — the only place the two stacks meet. Pre-tokenized parquet is read with
the trainer's own `ShardServer` (never streamed from HF).
"""

import argparse
from datetime import datetime
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import torch
from jax_single_pool.data import BatchSchedule, ShardServer, scan_shards
from jax_single_pool.load_run import HarvestForward, LoadedJaxRun, open_jax_run

from param_decomp.log import logger
from param_decomp_lab.harvest.accumulator import Harvester
from param_decomp_lab.harvest.config import HarvestConfig, ParamDecompHarvestConfig
from param_decomp_lab.harvest.repo import HarvestRepo
from param_decomp_lab.harvest.schemas import HarvestBatch, get_harvest_subrun_dir


def _to_torch(array: object) -> torch.Tensor:
    """Host a JAX/numpy array as a CPU torch tensor (copy: the accumulator's reservoir
    writes into the token windows in place)."""
    return torch.from_numpy(np.array(np.asarray(array)))


def harvest_batch_from_forward(
    tokens: np.ndarray, fwd: HarvestForward, activation_threshold: float
) -> HarvestBatch:
    """JAX forward outputs -> the torch `HarvestBatch` (matching the torch
    `ParamDecompHarvestFn`: lower-leaky CI as `causal_importance`, ‖U‖·(x@V) as
    `component_activation`, firing = CI > threshold)."""
    ci = {site: _to_torch(v) for site, v in fwd.lower_leaky_ci.items()}
    acts = {site: _to_torch(v) for site, v in fwd.component_acts.items()}
    return HarvestBatch(
        tokens=_to_torch(tokens).long(),  # torch harvest path is int64-keyed
        firings={site: ci[site] > activation_threshold for site in ci},
        activations={
            site: {"causal_importance": ci[site], "component_activation": acts[site]} for site in ci
        },
        output_probs=_to_torch(fwd.output_probs),
    )


def harvest_jax_run(
    run: LoadedJaxRun, config: HarvestConfig, activation_threshold: float, output_dir: Path
) -> None:
    data, seed = run.config.data, run.config.seed
    schedule = BatchSchedule(scan_shards(data.dir), config.batch_size, seed)
    server = ShardServer(schedule, data.seq_len, process_index=0, process_count=1)

    harvester = Harvester(
        layers=run.layer_activation_sizes,
        vocab_size=run.vocab_size,
        max_examples_per_component=config.activation_examples_per_component,
        context_tokens_per_side=config.activation_context_tokens_per_side,
        max_examples_per_batch_per_component=config.max_examples_per_batch_per_component,
        collect_component_cooccurrence=config.collect_component_cooccurrence,
        device=torch.device("cpu"),
    )

    assert isinstance(config.n_batches, int), "JAX harvest needs an explicit n_batches"
    for batch_idx in range(config.n_batches):
        tokens = server.local_batch(batch_idx)
        fwd = run.forward(jnp.asarray(tokens))
        hb = harvest_batch_from_forward(tokens, fwd, activation_threshold)
        harvester.process_batch(hb.tokens, hb.firings, hb.activations, hb.output_probs)
        logger.info(f"{batch_idx + 1}/{config.n_batches} batches")

    logger.info(
        f"Harvest complete: {config.n_batches} batches, {harvester.total_tokens_processed:,} tokens"
    )
    HarvestRepo.save_results(harvester, config, output_dir)
    logger.info(f"Saved results to {output_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run_dir", type=Path, required=True)
    ap.add_argument("--step", type=int, default=None, help="checkpoint step (default: latest)")
    ap.add_argument("--n_batches", type=int, required=True)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--activation_threshold", type=float, default=0.0)
    ap.add_argument("--subrun_id", type=str, default=None)
    ap.add_argument(
        "--no_cooccurrence",
        action="store_true",
        help="skip the O(C²) component-cooccurrence matrix (slow on CPU; on for production)",
    )
    args = ap.parse_args()

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
    logger.info(f"JAX harvest: run {run.run_id} step {run.step}, subrun {subrun_id}")
    harvest_jax_run(run, config, args.activation_threshold, output_dir)


if __name__ == "__main__":
    main()
