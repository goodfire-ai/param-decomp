from typing import Annotated

from pydantic import Field

from param_decomp.pretrain.models.gpt2 import GPT2, GPT2Config
from param_decomp.pretrain.models.gpt2_simple import GPT2Simple, GPT2SimpleConfig
from param_decomp.pretrain.models.llama import Llama, LlamaConfig
from param_decomp.pretrain.models.llama_simple import LlamaSimple, LlamaSimpleConfig
from param_decomp.pretrain.models.llama_simple_mlp import LlamaSimpleMLP, LlamaSimpleMLPConfig

# Discriminated union for model configs - Pydantic auto-selects based on model_type
ModelConfig = Annotated[
    GPT2Config | GPT2SimpleConfig | LlamaConfig | LlamaSimpleConfig | LlamaSimpleMLPConfig,
    Field(discriminator="model_type"),
]

# Mapping from model_type string to model class
MODEL_CLASSES: dict[str, type] = {
    "GPT2": GPT2,
    "GPT2Simple": GPT2Simple,
    "Llama": Llama,
    "LlamaSimple": LlamaSimple,
    "LlamaSimpleMLP": LlamaSimpleMLP,
}

__all__ = [
    "ModelConfig",
    "MODEL_CLASSES",
    "GPT2",
    "GPT2Config",
    "GPT2Simple",
    "GPT2SimpleConfig",
    "Llama",
    "LlamaConfig",
    "LlamaSimple",
    "LlamaSimpleConfig",
    "LlamaSimpleMLP",
    "LlamaSimpleMLPConfig",
]
