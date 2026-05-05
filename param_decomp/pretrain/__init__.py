"""Pretraining infrastructure for language models.

This module provides training scripts and model definitions for pretraining
language models that can later be decomposed using PD.

Usage:
    # Submit training job to SLURM
    pd-pretrain --config_path param_decomp/pretrain/configs/pile_llama_simple_mlp_4L.yaml

    # Run locally
    pd-pretrain --config_path ... --local

    # Multi-GPU training
    pd-pretrain --config_path ... --n_gpus 4

Available model types:
    - GPT2: Full GPT-2 implementation
    - GPT2Simple: Simplified GPT-2 (fewer optimizations)
    - Llama: Full Llama implementation
    - LlamaSimple: Simplified Llama (no QKV projection merging)
    - LlamaSimpleMLP: Llama with MLP-only architecture (for decomposition)

Output directory: PARAM_DECOMP_OUT_DIR/target_models/
"""

from param_decomp.pretrain.models import MODEL_CLASSES, ModelConfig
from param_decomp.pretrain.run_info import PretrainRunInfo

__all__ = ["PretrainRunInfo", "MODEL_CLASSES", "ModelConfig"]
