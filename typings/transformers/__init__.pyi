"""Use type stubs to speed up basedpyright calls."""

from pathlib import Path
from typing import Any, Self

from torch import nn
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

class PreTrainedModel(nn.Module):
    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: str | Path, /, *args: Any, **kwargs: Any
    ) -> Self: ...

class AutoTokenizer:
    # Actually differs from the original implementation which doesn't have a return type hint
    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: str | Path | None, /, *args: Any, **kwargs: Any
    ) -> PreTrainedTokenizerBase: ...

class LlamaForCausalLM(PreTrainedModel): ...
class GPT2LMHeadModel(PreTrainedModel): ...
class AutoModelForCausalLM(PreTrainedModel): ...
