"""Harvest CI-weighted component activation cooccurrence for interaction matrix H.

Computes H^l_{c,c'} as defined in the paper:

    H^l_{c,c'} = (Σ_{b,t} |g·a|_c · |g·a|_{c'}) / (Σ_{b,t} |g·a|_c²)

where g = CI lower_leaky mask, a = V^T @ x (component activation).

This is completely independent of param_decomp/harvest/. All outputs go to
param_decomp/scripts/geometric_interaction/out/<run_id>/interaction_harvest/.

Subcommands:
    worker  -- Run a single-GPU worker (optionally with rank/world_size)
    merge   -- Merge worker_states/*.pt into final data.pt
    submit  -- Submit SLURM array job (workers) + dependent merge job

Examples:
    # Single GPU
    python -m param_decomp.scripts.geometric_interaction.harvest_interaction worker \\
        --wandb_path="wandb:goodfire/spd/runs/s-55ea3f9b" --n_batches=5

    # Multi-GPU via SLURM
    python -m param_decomp.scripts.geometric_interaction.harvest_interaction submit \\
        --wandb_path="wandb:goodfire/spd/runs/s-55ea3f9b" --n_batches=20000 --n_gpus=8
"""

import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import einops
import fire
import torch
from jaxtyping import Float, Int
from pydantic import Field
from torch import Tensor

from param_decomp.base_config import BaseConfig
from param_decomp.log import logger
from param_decomp.utils.general_utils import bf16_autocast

if TYPE_CHECKING:
    from param_decomp.adapters.param_decomp import ParamDecompAdapter
    from param_decomp.configs import SamplingType
    from param_decomp.models.component_model import ComponentModel

SCRIPT_DIR = Path(__file__).parent


class InteractionHarvestConfig(BaseConfig):
    wandb_path: str = Field(..., description="wandb:entity/project/runs/run_id")
    n_batches: int = Field(20000, description="Total number of batches to process")
    batch_size: int = 32
    output_dir: str | None = None


def _resolve_output_dir(config: InteractionHarvestConfig, run_id: str) -> Path:
    if config.output_dir is not None:
        return Path(config.output_dir)
    return SCRIPT_DIR / "out" / run_id / "interaction_harvest"


def _extract_run_id(wandb_path: str) -> str:
    return wandb_path.rstrip("/").split("/")[-1]


# ── Accumulator ───────────────────────────────────────────────────────────────


class InteractionAccumulator:
    """Accumulates |g·a| pairwise statistics per module for H matrix computation.

    For each module with C components, maintains (in float64):
    - sum_ga_cross[C, C]: running Σ |g·a|_c · |g·a|_{c'}
    - total_tokens: count for normalization
    """

    def __init__(
        self,
        layers: list[tuple[str, int]],
        device: torch.device,
    ):
        self.layers = layers
        self.device = device
        self.total_tokens = 0

        self.sum_ga_cross: dict[str, Tensor] = {}
        for layer, c in layers:
            self.sum_ga_cross[layer] = torch.zeros(c, c, device=device, dtype=torch.float64)

    def process_batch(
        self,
        ga: dict[str, Float[Tensor, "B S C"]],
    ) -> None:
        for layer, ga_vals in ga.items():
            ga_flat = einops.rearrange(ga_vals, "b s c -> (b s) c").abs().double()
            self.total_tokens += ga_flat.shape[0]
            self.sum_ga_cross[layer] += einops.einsum(ga_flat, ga_flat, "S c1, S c2 -> c1 c2")

    def save(self, path: Path) -> None:
        data: dict[str, Any] = {
            "layers": self.layers,
            "total_tokens": self.total_tokens,
            "sum_ga_cross": {layer: m.cpu() for layer, m in self.sum_ga_cross.items()},
        }
        torch.save(data, path)

    @staticmethod
    def load(path: Path, device: torch.device) -> "InteractionAccumulator":
        data = torch.load(path, map_location="cpu", weights_only=False)
        acc = InteractionAccumulator(data["layers"], device)
        acc.total_tokens = data["total_tokens"]
        for layer, m in data["sum_ga_cross"].items():
            acc.sum_ga_cross[layer] = m.to(device)
        return acc

    def merge(self, other: "InteractionAccumulator") -> None:
        self.total_tokens += other.total_tokens
        for layer, _ in self.layers:
            self.sum_ga_cross[layer] += other.sum_ga_cross[layer].to(self.device)

    def build_results(self) -> dict[str, Any]:
        n_layers = len(self.layers)
        assert n_layers > 0
        tokens_per_layer = self.total_tokens // n_layers

        return {
            "sum_ga_cross": {layer: m.cpu() for layer, m in self.sum_ga_cross.items()},
            "total_tokens": tokens_per_layer,
            "layers": self.layers,
        }


# ── Model + dataloader ────────────────────────────────────────────────────────


def load_model_and_dataloader(
    wandb_path: str, batch_size: int
) -> tuple["ParamDecompAdapter", torch.utils.data.DataLoader[Tensor]]:
    from param_decomp.adapters.param_decomp import ParamDecompAdapter  # noqa: F811

    adapter = ParamDecompAdapter(wandb_path)
    dataloader = adapter.dataloader(batch_size)
    return adapter, dataloader


def compute_ga(
    model: "ComponentModel",
    batch: Int[Tensor, "B S"],
    sampling: "SamplingType",
) -> dict[str, Float[Tensor, "B S C"]]:
    """Forward pass → g·a per module (CI-weighted component activations)."""
    out = model(batch, cache_type="input")
    acts = model.get_all_component_acts(out.cache)
    ci = model.calc_causal_importances(out.cache, sampling=sampling, detach_inputs=True).lower_leaky
    return {layer: ci[layer] * acts[layer] for layer in model.target_module_paths}


# ── Worker ────────────────────────────────────────────────────────────────────


def worker(
    config_path: Path | str | None = None,
    *,
    config_json: dict[str, Any] | None = None,
    rank: int | None = None,
    world_size: int | None = None,
    **overrides: Any,
) -> None:
    """Run a single-GPU worker."""
    assert (rank is None) == (world_size is None)

    if config_path is not None:
        config = InteractionHarvestConfig.from_file(config_path)
    elif config_json is not None:
        config = InteractionHarvestConfig.model_validate(config_json)
    else:
        config = InteractionHarvestConfig(**overrides)

    run_id = _extract_run_id(config.wandb_path)
    output_dir = _resolve_output_dir(config, run_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Loading model from {config.wandb_path}")
    adapter, dataloader = load_model_and_dataloader(config.wandb_path, config.batch_size)
    model = adapter.component_model
    model.to(device).eval()
    sampling = adapter.pd_run_info.config.sampling

    layers = list(model.module_to_c.items())
    if rank is not None:
        logger.info(f"Worker rank={rank}/{world_size}")
    logger.info(f"Modules: {[(n, c) for n, c in layers]}")

    accumulator = InteractionAccumulator(layers, device)
    train_iter = iter(dataloader)
    t0 = time.time()
    batches_processed = 0

    for batch_idx in range(config.n_batches):
        try:
            batch_item = next(train_iter)
        except StopIteration:
            logger.info(f"Dataset exhausted after {batch_idx} batches")
            break

        if rank is not None and world_size is not None and batch_idx % world_size != rank:
            continue

        batch = batch_item.to(device)

        with torch.no_grad(), bf16_autocast():
            ga = compute_ga(model, batch, sampling)

        accumulator.process_batch(ga)
        batches_processed += 1

        if batches_processed % 100 == 0:
            elapsed = time.time() - t0
            rate = batches_processed / elapsed
            logger.info(
                f"  processed {batches_processed} batches "
                f"({rate:.1f} batch/s, {accumulator.total_tokens // len(layers):,} tokens)"
            )

    logger.info(f"Processed {batches_processed} batches total")

    if rank is not None:
        worker_dir = output_dir / "worker_states"
        worker_dir.mkdir(parents=True, exist_ok=True)
        worker_path = worker_dir / f"worker_{rank}.pt"
        accumulator.save(worker_path)
        logger.info(f"Saved worker state → {worker_path}")
    else:
        results = accumulator.build_results()
        results["config"] = config.model_dump()
        results["run_id"] = run_id
        torch.save(results, output_dir / "data.pt")
        logger.info(f"Saved → {output_dir / 'data.pt'}")
        _log_summary(results)


# ── Merge ─────────────────────────────────────────────────────────────────────


def merge(
    config_path: Path | str | None = None,
    *,
    config_json: dict[str, Any] | None = None,
    **overrides: Any,
) -> None:
    """Merge worker states into final data.pt."""
    if config_path is not None:
        config = InteractionHarvestConfig.from_file(config_path)
    elif config_json is not None:
        config = InteractionHarvestConfig.model_validate(config_json)
    else:
        config = InteractionHarvestConfig(**overrides)

    run_id = _extract_run_id(config.wandb_path)
    output_dir = _resolve_output_dir(config, run_id)
    worker_dir = output_dir / "worker_states"

    worker_files = sorted(worker_dir.glob("worker_*.pt"))
    assert worker_files, f"No worker states found in {worker_dir}"
    logger.info(f"Merging {len(worker_files)} worker states from {worker_dir}")

    device = torch.device("cpu")
    first, *rest = worker_files
    accumulator = InteractionAccumulator.load(first, device)
    logger.info(f"  loaded {first.name}")

    for wf in rest:
        other = InteractionAccumulator.load(wf, device)
        accumulator.merge(other)
        del other
        logger.info(f"  merged {wf.name}")

    results = accumulator.build_results()
    results["config"] = config.model_dump()
    results["run_id"] = run_id

    torch.save(results, output_dir / "data.pt")
    logger.info(f"Saved → {output_dir / 'data.pt'}")

    for wf in worker_files:
        wf.unlink()
    worker_dir.rmdir()
    logger.info("Cleaned up worker states")

    _log_summary(results)


# ── SLURM submission ──────────────────────────────────────────────────────────


def submit(
    config_path: Path | str | None = None,
    *,
    n_gpus: int = 8,
    time: str = "5:00:00",
    merge_time: str = "1:00:00",
    **overrides: Any,
) -> None:
    """Submit SLURM array job (workers) + dependent merge job."""
    from param_decomp.settings import DEFAULT_PARTITION_NAME
    from param_decomp.utils.slurm import (
        SlurmArrayConfig,
        SlurmConfig,
        generate_array_script,
        generate_script,
        submit_slurm_job,
    )

    if config_path is not None:
        config = InteractionHarvestConfig.from_file(config_path)
    else:
        config = InteractionHarvestConfig(**overrides)

    run_id = _extract_run_id(config.wandb_path)
    resolved_output_dir = str(_resolve_output_dir(config, run_id).resolve())
    config_dict = config.model_dump()
    config_dict["output_dir"] = resolved_output_dir
    config_json = InteractionHarvestConfig.model_validate(config_dict).model_dump_json()

    script_module = "param_decomp.scripts.geometric_interaction.harvest_interaction"

    worker_commands = []
    for rank in range(n_gpus):
        cmd = (
            f"python -m {script_module} worker"
            f" --config_json '{config_json}'"
            f" --rank={rank} --world_size={n_gpus}"
        )
        worker_commands.append(cmd)

    array_config = SlurmArrayConfig(
        job_name="int-harvest",
        partition=DEFAULT_PARTITION_NAME,
        n_gpus=1,
        time=time,
    )
    array_script = generate_array_script(array_config, worker_commands)
    array_result = submit_slurm_job(
        array_script, "int_harvest_worker", is_array=True, n_array_tasks=n_gpus
    )

    merge_cmd = f"python -m {script_module} merge --config_json '{config_json}'"
    merge_config = SlurmConfig(
        job_name="int-harvest-merge",
        partition=DEFAULT_PARTITION_NAME,
        n_gpus=0,
        time=merge_time,
        mem="64G",
        dependency_job_id=array_result.job_id,
    )
    merge_script = generate_script(merge_config, merge_cmd)
    merge_result = submit_slurm_job(merge_script, "int_harvest_merge")

    logger.info("Interaction harvest jobs submitted!")
    logger.info(f"  Array job: {array_result.job_id} ({n_gpus} workers)")
    logger.info(f"  Merge job: {merge_result.job_id} (depends on {array_result.job_id})")
    logger.info(f"  Worker logs: {array_result.log_pattern}")
    logger.info(f"  Merge log:  {merge_result.log_pattern}")
    logger.info(f"  Output: {resolved_output_dir}/data.pt")


# ── Helpers ───────────────────────────────────────────────────────────────────


def _log_summary(results: dict[str, Any]) -> None:
    n_tokens = results["total_tokens"]
    logger.info(f"Total tokens: {n_tokens:,}")
    for layer, cross in results["sum_ga_cross"].items():
        diag = cross.diag()
        n_alive = int((diag > 1e-6).sum().item())
        logger.info(f"  {layer}: {n_alive}/{len(diag)} components with nonzero |g·a|")


if __name__ == "__main__":
    fire.Fire({"worker": worker, "merge": merge, "submit": submit})
