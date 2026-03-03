"""Example: Harvest transcoder activations using the SPD harvest pipeline.

Loads trained BatchTopK transcoders from wandb artifacts and runs the generic
harvest pipeline to collect activation statistics (firing densities, token PMI,
activation examples).

Usage:
    python scripts/harvest_transcoders_example.py

Prerequisites:
    pip install -e ".[nn_decompositions]"
"""

from datetime import datetime

import torch

from spd.adapters.transcoder import TranscoderAdapter
from spd.harvest.config import HarvestConfig, TranscoderHarvestConfig
from spd.harvest.harvest import harvest
from spd.harvest.harvest_fn.transcoder import TranscoderHarvestFn
from spd.harvest.repo import HarvestRepo
from spd.harvest.schemas import get_harvest_subrun_dir

# -- Configuration ------------------------------------------------------------

# Fill in the wandb artifact paths for each layer's transcoder.
# Find these at: https://wandb.ai/mats-sprint/pile_transcoder_sweep3
# Each artifact should contain encoder.pt + config.json.
TRANSCODER_CONFIG = TranscoderHarvestConfig(
    base_model_path="wandb:goodfire/spd/t-32d1bb3b",
    artifact_paths={
        "h.0.mlp": "mats-sprint/pile_transcoder_sweep3/4096_batchtopk_k32_0.0003_L0_5d5b1f_checkpoint_final:v0",
        "h.1.mlp": "mats-sprint/pile_transcoder_sweep3/4096_batchtopk_k32_0.0003_L1_c208d7_checkpoint_final:v0",
        "h.2.mlp": "mats-sprint/pile_transcoder_sweep3/4096_batchtopk_k32_0.0003_L2_4f6e37_checkpoint_final:v0",
        "h.3.mlp": "mats-sprint/pile_transcoder_sweep3/4096_batchtopk_k32_0.0003_L3_e76468_checkpoint_final:v0",
    },
)

HARVEST_CONFIG = HarvestConfig(
    method_config=TRANSCODER_CONFIG,
    n_batches=20,
    batch_size=8,
    activation_examples_per_component=20,
    activation_context_tokens_per_side=10,
    pmi_token_top_k=10,
)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Build adapter (downloads artifacts + loads base model and transcoders)
    print("Loading transcoders and base model...")
    adapter = TranscoderAdapter(TRANSCODER_CONFIG)

    print(f"Base model vocab size: {adapter.vocab_size}")
    print(f"Layers: {adapter.layer_activation_sizes}")
    for path, tc in adapter.transcoders.items():
        print(
            f"  {path}: dict_size={tc.dict_size}, encoder_type={tc.cfg.encoder_type}, top_k={tc.cfg.top_k}"
        )

    # Build harvest function
    harvest_fn = TranscoderHarvestFn(adapter, TRANSCODER_CONFIG.activation_threshold, device)

    # Run harvest
    subrun_id = "h-" + datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = get_harvest_subrun_dir(adapter.decomposition_id, subrun_id)
    print(f"\nHarvesting to: {output_dir}")

    harvest(
        layers=adapter.layer_activation_sizes,
        vocab_size=adapter.vocab_size,
        dataloader=adapter.dataloader(HARVEST_CONFIG.batch_size),
        harvest_fn=harvest_fn,
        config=HARVEST_CONFIG,
        output_dir=output_dir,
        rank_world_size=None,
        device=device,
    )

    # Print summary
    print("\n=== Harvest Summary ===")
    repo = HarvestRepo(adapter.decomposition_id, subrun_id, readonly=True)
    components = repo.get_all_components()
    print(f"Total components harvested: {len(components)}")

    for comp in components[:10]:
        n_examples = len(comp.activation_examples)
        top_tokens = comp.input_token_pmi.top[:3]
        print(
            f"  {comp.component_key}: "
            f"density={comp.firing_density:.4f}, "
            f"examples={n_examples}, "
            f"mean_act={comp.mean_activations.get('activation', 0):.4f}, "
            f"top_pmi_tokens={top_tokens}"
        )

    if len(components) > 10:
        print(f"  ... and {len(components) - 10} more components")

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
