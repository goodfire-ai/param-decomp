"""Sanity check: verify BatchTopK firing counts match the expected budget."""

import torch
import yaml
from pydantic import TypeAdapter

from spd.adapters import adapter_from_config
from spd.harvest.config import DecompositionMethodHarvestConfig
from spd.harvest.harvest_fn import make_harvest_fn

CONFIGS = {
    "MSE TC": """
        type: TranscoderHarvestConfig
        base_model_path: "wandb:goodfire/spd/t-32d1bb3b"
        artifact_paths:
          "h.0.mlp": "mats-sprint/pile_transcoder_sweep3/4096_batchtopk_k32_0.0003_L0_5d5b1f_checkpoint_final:v0"
          "h.1.mlp": "mats-sprint/pile_transcoder_sweep3/4096_batchtopk_k32_0.0003_L1_c208d7_checkpoint_final:v0"
          "h.2.mlp": "mats-sprint/pile_transcoder_sweep3/4096_batchtopk_k32_0.0003_L2_4f6e37_checkpoint_final:v0"
          "h.3.mlp": "mats-sprint/pile_transcoder_sweep3/4096_batchtopk_k32_0.0003_L3_e76468_checkpoint_final:v0"
    """,
    "E2E TC": """
        type: TranscoderHarvestConfig
        base_model_path: "wandb:goodfire/spd/t-9d2b8f02"
        artifact_paths:
          "h.0.mlp": "mats-sprint/pile_e2e_sweep_jose/tc_parallel_k32_checkpoint_layer0_final:v0"
          "h.1.mlp": "mats-sprint/pile_e2e_sweep_jose/tc_parallel_k32_checkpoint_layer1_final:v0"
          "h.2.mlp": "mats-sprint/pile_e2e_sweep_jose/tc_parallel_k32_checkpoint_layer2_final:v0"
          "h.3.mlp": "mats-sprint/pile_e2e_sweep_jose/tc_parallel_k32_checkpoint_layer3_final:v0"
    """,
    "CLT": """
        type: CLTHarvestConfig
        base_model_path: "wandb:goodfire/spd/t-9d2b8f02"
        artifact_path: "mats-sprint/pile_e2e_sweep_jose/clt_parallel_k32_checkpoint_final:v0"
    """,
}

K = 32
BATCH_SIZE = 4
SEQ_LEN = 512
N_BATCHES = 5


def main():
    ta = TypeAdapter(DecompositionMethodHarvestConfig)
    device = torch.device("cpu")

    for name, raw_yaml in CONFIGS.items():
        method_config = ta.validate_python(yaml.safe_load(raw_yaml))
        adapter = adapter_from_config(method_config)
        harvest_fn = make_harvest_fn(device, method_config, adapter)
        loader = adapter.dataloader(batch_size=BATCH_SIZE)

        budget = K * BATCH_SIZE * SEQ_LEN
        print(f"\n{name} (id={method_config.id})")
        print(f"  Expected budget per layer: k={K} × B={BATCH_SIZE} × S={SEQ_LEN} = {budget}")

        for batch_idx, batch in enumerate(loader):
            if batch_idx >= N_BATCHES:
                break
            result = harvest_fn(batch)
            for path, firings in result.firings.items():
                nnz = firings.sum().item()
                delta = int(nnz - budget)
                status = "exact" if delta == 0 else f"short by {-delta}"
                print(f"  batch {batch_idx} | {path}: nnz={int(nnz):>6}  ({status})")


if __name__ == "__main__":
    main()
