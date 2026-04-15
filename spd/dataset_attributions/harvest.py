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
    harvest_subrun_id: str,
) -> dict[str, Bool[Tensor, " n_components"]]:
    """Build masks of alive components (firing_density > 0) per target layer.

    Only covers component layers — embed is always a valid source (not filtered).
    """

    component_alive = {
        layer: torch.zeros(model.module_to_c[layer], dtype=torch.bool)
        for layer in model.target_module_paths
    }

    harvest = HarvestRepo(decomposition_id=run_id, subrun_id=harvest_subrun_id, readonly=True)

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
    harvest_subrun_id: str,
    rank: int,
    world_size: int,
) -> None:
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
    component_alive = _build_alive_masks(model, run_id, harvest_subrun_id)

    harvester = AttributionHarvester(
        model=model,
        topology=topology,
        sources_by_target=sources_by_target,
        component_alive=component_alive,
        sampling=spd_config.sampling,
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

        if batch_idx % world_size != rank:
            continue

        batch = extract_batch_data(batch_data).to(device)
        harvester.process_batch(batch)

    logger.info(f"Processing complete. Tokens: {harvester.n_tokens:,}")

    storage = harvester.finalize(config.ci_threshold)

    worker_dir = output_dir / "worker_states"
    worker_dir.mkdir(parents=True, exist_ok=True)
    output_path = worker_dir / f"dataset_attributions_rank_{rank}.pt"
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
