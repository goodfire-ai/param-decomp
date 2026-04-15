"""Standalone harvest for activation-based cooccurrence statistics.

Runs forward passes over training data and accumulates:
1. Binary coactivation at multiple activation-magnitude thresholds
2. Continuous pairwise Pearson correlation of component activations

This is completely independent of the main harvest pipeline in spd/harvest/.
All outputs go to spd/scripts/geometric_interaction/out/<run_id>/activation_cooccurrence/.
Nothing is written to SPD_OUT_DIR/harvest/.

Subcommands:
    worker  -- Run a single-GPU worker (optionally with rank/world_size for multi-GPU)
    merge   -- Merge worker_states/*.pt into final data.pt
    submit  -- Submit SLURM array job (workers) + dependent merge job

Examples:
    # Single GPU (no SLURM)
    python -m spd.scripts.geometric_interaction.harvest_activations worker \\
        --wandb_path="wandb:goodfire/spd/runs/s-55ea3f9b" --n_batches=20000

    # Multi-GPU via SLURM
    python -m spd.scripts.geometric_interaction.harvest_activations submit \\
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

from spd.base_config import BaseConfig
from spd.log import logger
from spd.models.component_model import ComponentModel
from spd.utils.general_utils import bf16_autocast, extract_batch_data

if TYPE_CHECKING:
    from spd.adapters.spd import SPDAdapter

SCRIPT_DIR = Path(__file__).parent


class ActivationHarvestConfig(BaseConfig):
    wandb_path: str = Field(..., description="wandb:entity/project/runs/run_id")
    n_batches: int = Field(20000, description="Total number of batches to process")
    batch_size: int = 32
    activation_thresholds: list[float] = Field(
        default=[0.0, 0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0],
        description="Thresholds on |component_activation| for binary coactivation",
    )
    output_dir: str | None = None


def _resolve_output_dir(config: ActivationHarvestConfig, run_id: str) -> Path:
    if config.output_dir is not None:
        return Path(config.output_dir)
    return SCRIPT_DIR / "out" / run_id / "activation_cooccurrence"


def _extract_run_id(wandb_path: str) -> str:
    return wandb_path.rstrip("/").split("/")[-1]


# ── Accumulator ───────────────────────────────────────────────────────────────


class PairwiseAccumulator:
    """Accumulates pairwise statistics per module on GPU.

    For each module with C components, maintains:
    - Binary coactivation counts at each threshold: count_i[t][C], count_ij[t][C,C]
    - Continuous correlation running sums: sum_a[C], sum_a2[C], sum_ab[C,C]
    - Total token count
    """

    def __init__(
        self,
        layers: list[tuple[str, int]],
        thresholds: list[float],
        device: torch.device,
    ):
        self.layers = layers
        self.thresholds = thresholds
        self.device = device
        self.total_tokens = 0

        # Binary coactivation: {threshold: {layer: (count_i, count_ij)}}
        self.binary: dict[float, dict[str, tuple[Tensor, Tensor]]] = {}
        for t in thresholds:
            self.binary[t] = {}
            for layer, c in layers:
                count_i = torch.zeros(c, device=device, dtype=torch.float64)
                count_ij = torch.zeros(c, c, device=device, dtype=torch.float64)
                self.binary[t][layer] = (count_i, count_ij)

        # Continuous correlation: {layer: (sum_a, sum_a2, sum_ab)}
        self.continuous: dict[str, tuple[Tensor, Tensor, Tensor]] = {}
        for layer, c in layers:
            sum_a = torch.zeros(c, device=device, dtype=torch.float64)
            sum_a2 = torch.zeros(c, device=device, dtype=torch.float64)
            sum_ab = torch.zeros(c, c, device=device, dtype=torch.float64)
            self.continuous[layer] = (sum_a, sum_a2, sum_ab)

    def process_batch(
        self,
        activations: dict[str, Float[Tensor, "B S C"]],
    ) -> None:
        for layer, act in activations.items():
            act_flat = einops.rearrange(act, "b s c -> (b s) c")
            n_tokens = act_flat.shape[0]
            self.total_tokens += n_tokens

            act_abs = act_flat.abs()
            act_f64 = act_abs.double()

            # Binary coactivation at each threshold
            for t in self.thresholds:
                firing = (act_abs > t).float()
                count_i, count_ij = self.binary[t][layer]
                count_i += firing.sum(dim=0).double()
                count_ij += einops.einsum(firing.double(), firing.double(), "S c1, S c2 -> c1 c2")

            # Continuous sums for Pearson correlation
            sum_a, sum_a2, sum_ab = self.continuous[layer]
            sum_a += act_f64.sum(dim=0)
            sum_a2 += (act_f64**2).sum(dim=0)
            sum_ab += einops.einsum(act_f64, act_f64, "S c1, S c2 -> c1 c2")

    def save(self, path: Path) -> None:
        data: dict[str, Any] = {
            "layers": self.layers,
            "thresholds": self.thresholds,
            "total_tokens": self.total_tokens,
            "binary": {
                t: {layer: (ci.cpu(), cij.cpu()) for layer, (ci, cij) in layer_dict.items()}
                for t, layer_dict in self.binary.items()
            },
            "continuous": {
                layer: (sa.cpu(), sa2.cpu(), sab.cpu())
                for layer, (sa, sa2, sab) in self.continuous.items()
            },
        }
        torch.save(data, path)

    @staticmethod
    def load(path: Path, device: torch.device) -> "PairwiseAccumulator":
        data = torch.load(path, map_location="cpu", weights_only=False)
        acc = PairwiseAccumulator(data["layers"], data["thresholds"], device)
        acc.total_tokens = data["total_tokens"]
        for t, layer_dict in data["binary"].items():
            for layer, (ci, cij) in layer_dict.items():
                acc.binary[t][layer] = (ci.to(device), cij.to(device))
        for layer, (sa, sa2, sab) in data["continuous"].items():
            acc.continuous[layer] = (sa.to(device), sa2.to(device), sab.to(device))
        return acc

    def merge(self, other: "PairwiseAccumulator") -> None:
        self.total_tokens += other.total_tokens
        for t in self.thresholds:
            for layer, _ in self.layers:
                self_ci, self_cij = self.binary[t][layer]
                other_ci, other_cij = other.binary[t][layer]
                self_ci += other_ci.to(self_ci.device)
                self_cij += other_cij.to(self_cij.device)
        for layer, _ in self.layers:
            self_sa, self_sa2, self_sab = self.continuous[layer]
            other_sa, other_sa2, other_sab = other.continuous[layer]
            self_sa += other_sa.to(self_sa.device)
            self_sa2 += other_sa2.to(self_sa2.device)
            self_sab += other_sab.to(self_sab.device)

    def build_results(self) -> dict[str, Any]:
        # total_tokens is accumulated once per layer per batch; correct for that.
        n_layers = len(self.layers)
        assert n_layers > 0
        tokens_per_layer = self.total_tokens // n_layers

        results: dict[str, Any] = {"total_tokens": tokens_per_layer}

        # Binary coactivation fractions
        binary_results: dict[float, dict[str, dict[str, Tensor]]] = {}
        for t in self.thresholds:
            binary_results[t] = {}
            for layer, _ in self.layers:
                count_i, count_ij = self.binary[t][layer]
                ci = count_i.cpu().float()
                cij = count_ij.cpu().float()
                # P(i active | j active) = count_ij[i,j] / count_j
                denom = ci.unsqueeze(0).expand_as(cij)
                frac = cij / denom
                frac = torch.nan_to_num(frac, nan=0.0)
                binary_results[t][layer] = {
                    "count_i": ci,
                    "count_ij": cij,
                    "coactivation_fraction": frac,
                    "activation_density": ci / tokens_per_layer,
                }
        results["binary"] = binary_results

        # Pearson correlations
        n = float(tokens_per_layer)
        pearson_results: dict[str, Tensor] = {}
        for layer, _ in self.layers:
            sum_a, sum_a2, sum_ab = self.continuous[layer]
            sa = sum_a.cpu().double()
            sa2 = sum_a2.cpu().double()
            sab = sum_ab.cpu().double()

            mean_a = sa / n
            var_a = sa2 / n - mean_a**2
            std_a = var_a.clamp(min=1e-30).sqrt()

            # cov(i,j) = E[a_i * a_j] - E[a_i] * E[a_j]
            cov = sab / n - einops.einsum(mean_a, mean_a, "c1, c2 -> c1 c2")
            # corr(i,j) = cov(i,j) / (std_i * std_j)
            denom = einops.einsum(std_a, std_a, "c1, c2 -> c1 c2")
            corr = cov / denom
            corr = torch.nan_to_num(corr, nan=0.0).float()
            pearson_results[layer] = corr

        results["pearson_correlation"] = pearson_results
        results["continuous_stats"] = {
            layer: {
                "sum_a": sum_a.cpu(),
                "sum_a2": sum_a2.cpu(),
                "sum_ab": sum_ab.cpu(),
            }
            for layer, (sum_a, sum_a2, sum_ab) in self.continuous.items()
        }
        return results


# ── Model + dataloader loading ────────────────────────────────────────────────


def load_model_and_dataloader(
    wandb_path: str, batch_size: int
) -> tuple["SPDAdapter", torch.utils.data.DataLoader[Any]]:
    from spd.adapters.spd import SPDAdapter  # noqa: F811
    from spd.utils.wandb_utils import parse_wandb_run_path

    _, _, run_id = parse_wandb_run_path(wandb_path)
    adapter = SPDAdapter(run_id)
    dataloader = adapter.dataloader(batch_size)
    return adapter, dataloader


def compute_component_activations(
    model: "ComponentModel",
    batch: Int[Tensor, "B S"],
) -> dict[str, Float[Tensor, "B S C"]]:
    """Run forward pass and return per-layer component activations (V^T x * ||U||)."""
    out = model(batch, cache_type="input")
    per_layer_acts = model.get_all_component_acts(out.cache)
    u_norms = {
        layer_name: component.U.norm(dim=1) for layer_name, component in model.components.items()
    }
    return {layer: per_layer_acts[layer] * u_norms[layer] for layer in model.target_module_paths}


# ── Worker ────────────────────────────────────────────────────────────────────


def worker(
    config_path: Path | str | None = None,
    *,
    config_json: dict[str, Any] | None = None,
    rank: int | None = None,
    world_size: int | None = None,
    **overrides: Any,
) -> None:
    """Run a single-GPU worker. With rank/world_size, processes a subset of batches."""
    assert (rank is None) == (world_size is None), (
        "rank and world_size must both be set or both None"
    )

    if config_path is not None:
        config = ActivationHarvestConfig.from_file(config_path)
    elif config_json is not None:
        config = ActivationHarvestConfig.model_validate(config_json)
    else:
        config = ActivationHarvestConfig(**overrides)

    run_id = _extract_run_id(config.wandb_path)
    output_dir = _resolve_output_dir(config, run_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Loading model from {config.wandb_path}")
    adapter, dataloader = load_model_and_dataloader(config.wandb_path, config.batch_size)
    model = adapter.component_model
    model.to(device).eval()

    layers = list(model.module_to_c.items())

    if rank is not None:
        logger.info(f"Worker rank={rank}/{world_size}")
    logger.info(f"Modules: {[(n, c) for n, c in layers]}")
    logger.info(f"Thresholds: {config.activation_thresholds}")

    accumulator = PairwiseAccumulator(layers, config.activation_thresholds, device)

    train_iter = iter(dataloader)
    t0 = time.time()
    batches_processed = 0

    for batch_idx in range(config.n_batches):
        try:
            batch_item = next(train_iter)
        except StopIteration:
            logger.info(f"Dataset exhausted after {batch_idx} batches")
            break

        # Modulo distribution: skip batches not assigned to this rank
        if rank is not None and world_size is not None and batch_idx % world_size != rank:
            continue

        batch = extract_batch_data(batch_item).to(device)

        with torch.no_grad(), bf16_autocast():
            activations = compute_component_activations(model, batch)

        accumulator.process_batch(activations)
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
        # Multi-GPU: save worker state for later merge
        worker_dir = output_dir / "worker_states"
        worker_dir.mkdir(parents=True, exist_ok=True)
        worker_path = worker_dir / f"worker_{rank}.pt"
        accumulator.save(worker_path)
        logger.info(f"Saved worker state → {worker_path}")
    else:
        # Single-GPU: save final results directly
        results = accumulator.build_results()
        results["config"] = config.model_dump()
        results["run_id"] = run_id
        data_path = output_dir / "data.pt"
        torch.save(results, data_path)
        logger.info(f"Saved → {data_path}")
        _log_summary(results, layers, config.activation_thresholds)


# ── Merge ─────────────────────────────────────────────────────────────────────


def merge(
    config_path: Path | str | None = None,
    *,
    config_json: dict[str, Any] | None = None,
    **overrides: Any,
) -> None:
    """Merge worker states into final data.pt."""
    if config_path is not None:
        config = ActivationHarvestConfig.from_file(config_path)
    elif config_json is not None:
        config = ActivationHarvestConfig.model_validate(config_json)
    else:
        config = ActivationHarvestConfig(**overrides)

    run_id = _extract_run_id(config.wandb_path)
    output_dir = _resolve_output_dir(config, run_id)
    worker_dir = output_dir / "worker_states"

    worker_files = sorted(worker_dir.glob("worker_*.pt"))
    assert worker_files, f"No worker states found in {worker_dir}"
    logger.info(f"Merging {len(worker_files)} worker states from {worker_dir}")

    device = torch.device("cpu")
    first, *rest = worker_files
    accumulator = PairwiseAccumulator.load(first, device)
    logger.info(f"  loaded {first.name}")

    for wf in rest:
        other = PairwiseAccumulator.load(wf, device)
        accumulator.merge(other)
        del other
        logger.info(f"  merged {wf.name}")

    results = accumulator.build_results()
    results["config"] = config.model_dump()
    results["run_id"] = run_id

    data_path = output_dir / "data.pt"
    torch.save(results, data_path)
    logger.info(f"Saved → {data_path}")

    # Clean up worker states
    for wf in worker_files:
        wf.unlink()
    worker_dir.rmdir()
    logger.info("Cleaned up worker states")

    _log_summary(results, accumulator.layers, config.activation_thresholds)


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
    from spd.settings import DEFAULT_PARTITION_NAME
    from spd.utils.slurm import (
        SlurmArrayConfig,
        SlurmConfig,
        generate_array_script,
        generate_script,
        submit_slurm_job,
    )

    if config_path is not None:
        config = ActivationHarvestConfig.from_file(config_path)
    else:
        config = ActivationHarvestConfig(**overrides)

    # Resolve output_dir to an absolute path before serializing for workers.
    # Workers run from a different cwd, so relative/None paths won't resolve correctly.
    run_id = _extract_run_id(config.wandb_path)
    resolved_output_dir = str(_resolve_output_dir(config, run_id).resolve())
    config_dict = config.model_dump()
    config_dict["output_dir"] = resolved_output_dir
    config_json = ActivationHarvestConfig.model_validate(config_dict).model_dump_json()

    script_module = "spd.scripts.geometric_interaction.harvest_activations"

    # Worker commands: one per GPU rank
    worker_commands = []
    for rank in range(n_gpus):
        cmd = (
            f"python -m {script_module} worker"
            f" --config_json '{config_json}'"
            f" --rank={rank} --world_size={n_gpus}"
        )
        worker_commands.append(cmd)

    array_config = SlurmArrayConfig(
        job_name="act-harvest",
        partition=DEFAULT_PARTITION_NAME,
        n_gpus=1,
        time=time,
    )
    array_script = generate_array_script(array_config, worker_commands)
    array_result = submit_slurm_job(
        array_script, "act_harvest_worker", is_array=True, n_array_tasks=n_gpus
    )

    # Merge job: CPU-only, depends on array completion
    merge_cmd = f"python -m {script_module} merge --config_json '{config_json}'"
    merge_config = SlurmConfig(
        job_name="act-harvest-merge",
        partition=DEFAULT_PARTITION_NAME,
        n_gpus=0,
        time=merge_time,
        mem="64G",
        dependency_job_id=array_result.job_id,
    )
    merge_script = generate_script(merge_config, merge_cmd)
    merge_result = submit_slurm_job(merge_script, "act_harvest_merge")

    run_id = _extract_run_id(config.wandb_path)
    output_dir = _resolve_output_dir(config, run_id)

    logger.info("Activation harvest jobs submitted!")
    logger.info(f"  Array job: {array_result.job_id} ({n_gpus} workers)")
    logger.info(f"  Merge job: {merge_result.job_id} (depends on {array_result.job_id})")
    logger.info(f"  Worker logs: {array_result.log_pattern}")
    logger.info(f"  Merge log:  {merge_result.log_pattern}")
    logger.info(f"  Output: {output_dir}/data.pt")


# ── Helpers ───────────────────────────────────────────────────────────────────


def _log_summary(
    results: dict[str, Any],
    layers: list[tuple[str, int]],
    thresholds: list[float],
) -> None:
    n_tokens = results["total_tokens"]
    logger.info(f"Total tokens: {n_tokens:,}")
    for t in thresholds:
        for layer, _ in layers:
            density = results["binary"][t][layer]["activation_density"]
            n_alive = int((density > 0.001).sum().item())
            logger.info(f"  threshold={t}, {layer}: {n_alive}/{len(density)} alive (>0.1% density)")


if __name__ == "__main__":
    fire.Fire({"worker": worker, "merge": merge, "submit": submit})
