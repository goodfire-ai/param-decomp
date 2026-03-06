"""Dataset attribution harvesting.

Computes component-to-component attribution strengths aggregated over the full
training dataset. Unlike prompt attributions (single-prompt, position-aware),
dataset attributions answer: "In aggregate, which components typically influence
each other?"

Uses residual-based storage for scalability:
- Component targets: stored directly
- Output targets: stored as attributions to output residual, computed on-the-fly at query time

See CLAUDE.md in this directory for usage instructions.
"""

import itertools
from pathlib import Path

import torch
import tqdm
from jaxtyping import Bool
from torch import Tensor

from spd.data import train_loader_and_tokenizer
from spd.dataset_attributions.config import DatasetAttributionConfig
from spd.dataset_attributions.harvester import AttributionHarvester
from spd.dataset_attributions.storage import DatasetAttributionStorage
from spd.harvest.repo import HarvestRepo
from spd.log import logger
from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.topology import TransformerTopology, get_sources_by_target
from spd.utils.distributed_utils import get_device
from spd.utils.general_utils import extract_batch_data
from spd.utils.wandb_utils import parse_wandb_run_path


def _build_alive_masks(
    model: ComponentModel,
    run_id: str,
    harvest_subrun_id: str | None,
    embed_path: str,
    vocab_size: int,
) -> dict[str, Bool[Tensor, " n_components"]]:
    """Build masks of alive components (mean_activation > threshold) for sources and targets.

    Falls back to all-alive if harvest summary not available.

    Index structure:
    - Sources: [0, vocab_size) = wte tokens, [vocab_size, vocab_size + n_components) = component layers
    - Targets: [0, n_components) = component layers (output handled via out_residual)
    """

    component_alive = {
        embed_path: torch.ones(vocab_size, dtype=torch.bool),  # TODO(oli): maybe remove this
        **{
            layer: torch.zeros(model.module_to_c[layer], dtype=torch.bool)
            for layer in model.target_module_paths
        },
    }

    if harvest_subrun_id is not None:
        harvest = HarvestRepo(decomposition_id=run_id, subrun_id=harvest_subrun_id, readonly=True)
    else:
        harvest = HarvestRepo.open_most_recent(run_id, readonly=True)
        assert harvest is not None, f"No harvest data for {run_id}"

    summary = harvest.get_summary()
    assert summary is not None, "Harvest summary not available"

    for layer in model.target_module_paths:
        n_layer_components = model.module_to_c[layer]
        for c_idx in range(n_layer_components):
            component_key = f"{layer}:{c_idx}"
            is_alive = component_key in summary and summary[component_key].firing_density > 0.0
            component_alive[layer][c_idx] = is_alive

    return component_alive


def harvest_attributions(
    config: DatasetAttributionConfig,
    output_dir: Path,
    harvest_subrun_id: str | None = None,
    rank: int | None = None,
    world_size: int | None = None,
) -> None:
    """Compute dataset attributions over the training dataset.

    Args:
        wandb_path: WandB run path for the target decomposition run.
        config: Configuration for attribution harvesting.
        output_dir: Directory to write results into.
        harvest_subrun_id: Harvest subrun to use for alive masks. If None, uses most recent.
        rank: Worker rank for parallel execution (0 to world_size-1).
        world_size: Total number of workers. If specified with rank, only processes
            batches where batch_idx % world_size == rank.
    """

    assert (rank is None) == (world_size is None), "rank and world_size must both be set or unset"

    device = torch.device(get_device())
    logger.info(f"Loading model on {device}")

    _, _, run_id = parse_wandb_run_path(config.spd_run_wandb_path)

    run_info = SPDRunInfo.from_path(config.spd_run_wandb_path)
    model = ComponentModel.from_run_info(run_info).to(device)
    model.eval()

    spd_config = run_info.config
    train_loader, _ = train_loader_and_tokenizer(spd_config, config.batch_size)

    # Get gradient connectivity
    logger.info("Computing sources_by_target...")
    topology = TransformerTopology(model.target_model)
    embed_path = topology.path_schema.embedding_path
    unembed_path = topology.path_schema.unembed_path
    vocab_size = topology.embedding_module.num_embeddings
    logger.info(f"Vocab size: {vocab_size}")
    sources_by_target_raw = get_sources_by_target(model, topology, str(device), spd_config.sampling)

    # Filter to valid source/target pairs:
    # - Valid sources: embedding + component layers
    # - Valid targets: component layers + unembed
    component_layers = set(model.target_module_paths)
    valid_sources = component_layers | {embed_path}
    valid_targets = component_layers | {unembed_path}

    sources_by_target: dict[str, list[str]] = {}
    for target, sources in sources_by_target_raw.items():
        if target not in valid_targets:
            continue
        filtered_sources = [src for src in sources if src in valid_sources]
        if filtered_sources:
            sources_by_target[target] = filtered_sources
    logger.info(f"Found {len(sources_by_target)} target layers with gradient connections")

    # Build alive masks
    component_alive = _build_alive_masks(model, run_id, harvest_subrun_id, embed_path, vocab_size)

    # Create harvester (all concrete paths internally)
    harvester = AttributionHarvester(
        model=model,
        sources_by_target=sources_by_target,
        vocab_size=vocab_size,
        component_alive=component_alive,
        sampling=spd_config.sampling,
        embed_path=embed_path,
        embedding_module=topology.embedding_module,
        unembed_path=unembed_path,
        unembed_module=topology.unembed_module,
        device=device,
    )

    # Process batches
    train_iter = iter(train_loader)
    match config.n_batches:
        case int(n_batches):
            batch_range = range(n_batches)
        case "whole_dataset":
            batch_range = itertools.count()

    for batch_idx in tqdm.tqdm(batch_range, desc="Attribution batches"):
        try:
            batch_data = next(train_iter)
        except StopIteration:
            logger.info(f"Dataset exhausted at batch {batch_idx}. Processing complete.")
            break

        # Skip batches not assigned to this rank
        if world_size is not None and batch_idx % world_size != rank:
            continue

        batch = extract_batch_data(batch_data).to(device)
        harvester.process_batch(batch)

    logger.info(f"Processing complete. Tokens: {harvester.n_tokens:,}")

    storage = harvester.finalize(topology, config.ci_threshold)

    if rank is not None:
        worker_dir = output_dir / "worker_states"
        worker_dir.mkdir(parents=True, exist_ok=True)
        output_path = worker_dir / f"dataset_attributions_rank_{rank}.pt"
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "dataset_attributions.pt"
    storage.save(output_path)


def merge_attributions(output_dir: Path) -> None:
    """Merge partial attribution files from parallel workers."""
    worker_dir = output_dir / "worker_states"
    rank_files = sorted(worker_dir.glob("dataset_attributions_rank_*.pt"))
    assert rank_files, f"No rank files found in {worker_dir}"
    logger.info(f"Found {len(rank_files)} rank files to merge")

    merged = DatasetAttributionStorage.merge(rank_files)

    output_path = output_dir / "dataset_attributions.pt"
    merged.save(output_path)
    logger.info(f"Total: {merged.n_tokens_processed:,} tokens")

    # TODO(oli): reenable this
    # disabled deletion for testing, posterity and retries
    # for rank_file in rank_files:
    #     rank_file.unlink()
    # worker_dir.rmdir()
    # logger.info(f"Deleted {len(rank_files)} per-rank files and worker_states/")
