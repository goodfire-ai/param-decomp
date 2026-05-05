# param_decomp/pretrain - Language Model Pretraining

This module provides infrastructure for pretraining language models that can
later be decomposed using PD.

## Overview

- **Purpose**: Train GPT-2 and Llama variants on SimpleStories or Pile datasets
- **Output**: Models saved to `PARAM_DECOMP_OUT_DIR/target_models/`
- **CLI**: `pd-pretrain`

## CLI Usage

```bash
# Submit to SLURM (default)
pd-pretrain --config_path param_decomp/pretrain/configs/pile_llama_simple_mlp_4L.yaml

# Run locally
pd-pretrain --config_path ... --local

# Multi-GPU DDP training
pd-pretrain --config_path ... --n_gpus 4

```

## Available Models

| Model Type | Description |
|------------|-------------|
| `GPT2` | Full GPT-2 implementation |
| `GPT2Simple` | Simplified GPT-2 |
| `Llama` | Full Llama implementation |
| `LlamaSimple` | Simplified Llama (no QKV merging) |
| `LlamaSimpleMLP` | Llama MLP-only (primary decomposition target) |

## Tokenizers

- **SimpleStories**: `SimpleStories/test-SimpleStories-gpt2-1.25M` (vocab size: 4019)
- **Pile/OpenWebText**: `gpt2` (vocab size: 50257)

## Dataset n_ctx vs Model n_ctx

The dataset `n_ctx` must be **model n_ctx + 1**. During training, sequences are split into
input `[:, :-1]` and target `[:, 1:]` for next-token prediction, so the extra token provides
room for label indexing. For example, if the model has `n_ctx: 512`, the dataset should have
`n_ctx: 513`.

## Key Files

- `train.py` - Main training loop with DDP support
- `run_info.py` - Load trained models from W&B or local paths
- `models/` - Model implementations
- `configs/` - Training configuration YAML files
- `scripts/run_slurm_cli.py` - CLI entry point
- `scripts/run_slurm.py` - SLURM submission and local run logic

## Loading Trained Models

```python
from param_decomp.pretrain.run_info import PretrainRunInfo
from param_decomp.pretrain.models import MODEL_CLASSES

# Load from W&B
run_info = PretrainRunInfo.from_path("wandb:entity/project/runs/run_id")
model_cls = MODEL_CLASSES[run_info.model_config_dict["model_type"]]
model = model_cls.from_run_info(run_info)

# Load from local path
run_info = PretrainRunInfo.from_path("/path/to/checkpoints/model_step_10000.pt")
```

Downloaded W&B runs are cached in `PARAM_DECOMP_OUT_DIR/pretrain_cache/<project>-<run_id>/`.

## Output Structure

```
PARAM_DECOMP_OUT_DIR/target_models/
└── <timestamp>/
    ├── final_config.yaml      # Full training config
    ├── model_config.yaml      # Model architecture config
    ├── tokenizer.json         # Tokenizer (uploaded to W&B)
    ├── main.log               # Training log
    └── checkpoints/
        ├── model_step_0.pt
        ├── model_step_1.pt
        ├── model_step_2.pt
        ├── model_step_4.pt    # Power-of-2 checkpoints
        └── ...
```

## Integration with PD

After training, models can be decomposed using PD:

```yaml
# In param_decomp/experiments/lm/*.yaml
pretrained_model_class: param_decomp.pretrain.models.llama_simple_mlp.LlamaSimpleMLP
pretrained_model_name: wandb:goodfire/spd/runs/<run_id>
```
