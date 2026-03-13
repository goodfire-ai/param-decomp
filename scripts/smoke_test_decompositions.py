"""Smoke test: load each decomposition type, run one batch through the harvest fn."""

import torch
import yaml

from spd.adapters import adapter_from_config
from spd.harvest.config import DecompositionMethodHarvestConfig
from spd.harvest.harvest_fn import make_harvest_fn

CONFIGS = {
    "MSE TC (pile_transcoder_sweep3, k32, base=t-32d1bb3b)": """
        type: TranscoderHarvestConfig
        base_model_path: "wandb:goodfire/spd/t-32d1bb3b"
        artifact_paths:
          "h.0.mlp": "mats-sprint/pile_transcoder_sweep3/4096_batchtopk_k32_0.0003_L0_5d5b1f_checkpoint_final:v0"
          "h.1.mlp": "mats-sprint/pile_transcoder_sweep3/4096_batchtopk_k32_0.0003_L1_c208d7_checkpoint_final:v0"
          "h.2.mlp": "mats-sprint/pile_transcoder_sweep3/4096_batchtopk_k32_0.0003_L2_4f6e37_checkpoint_final:v0"
          "h.3.mlp": "mats-sprint/pile_transcoder_sweep3/4096_batchtopk_k32_0.0003_L3_e76468_checkpoint_final:v0"
    """,
    "E2E TC (pile_e2e_sweep_jose, k32, base=t-9d2b8f02)": """
        type: TranscoderHarvestConfig
        base_model_path: "wandb:goodfire/spd/t-9d2b8f02"
        artifact_paths:
          "h.0.mlp": "mats-sprint/pile_e2e_sweep_jose/tc_parallel_k32_checkpoint_layer0_final:v0"
          "h.1.mlp": "mats-sprint/pile_e2e_sweep_jose/tc_parallel_k32_checkpoint_layer1_final:v0"
          "h.2.mlp": "mats-sprint/pile_e2e_sweep_jose/tc_parallel_k32_checkpoint_layer2_final:v0"
          "h.3.mlp": "mats-sprint/pile_e2e_sweep_jose/tc_parallel_k32_checkpoint_layer3_final:v0"
    """,
    "CLT (pile_e2e_sweep_jose, parallel k32, base=t-9d2b8f02)": """
        type: CLTHarvestConfig
        base_model_path: "wandb:goodfire/spd/t-9d2b8f02"
        artifact_path: "mats-sprint/pile_e2e_sweep_jose/clt_parallel_k32_checkpoint_final:v0"
    """,
}


def main():
    from pydantic import TypeAdapter

    ta = TypeAdapter(DecompositionMethodHarvestConfig)
    device = torch.device("cpu")

    for name, raw_yaml in CONFIGS.items():
        print(f"\n{'=' * 60}")
        print(f"  {name}")
        print(f"{'=' * 60}")

        method_config = ta.validate_python(yaml.safe_load(raw_yaml))
        print(f"  ID: {method_config.id}")

        adapter = adapter_from_config(method_config)
        print(f"  Layers: {adapter.layer_activation_sizes}")

        harvest_fn = make_harvest_fn(device, method_config, adapter)
        batch = next(iter(adapter.dataloader(batch_size=2)))
        result = harvest_fn(batch)

        print(f"  Tokens: {result.tokens.shape}")
        for path, firings in result.firings.items():
            nnz = firings.sum().item()
            print(f"  {path}: firings={firings.shape}, nnz={nnz:.0f}")

        print("  OK")

    print(f"\n{'=' * 60}")
    print("  All decomposition types passed.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
