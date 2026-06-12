"""Use type stubs to speed up basedpyright calls."""

from pathlib import Path
from typing import Any, Self

import numpy as np
from numpy.typing import NDArray
from torch import nn

class PreTrainedModel(nn.Module):
    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: str | Path, /, *args: Any, **kwargs: Any
    ) -> Self: ...

class PreTrainedTokenizer:
    bos_token_id: int
    eos_token: str

    def encode(self, text: str, /, *args: Any, **kwargs: Any) -> NDArray[np.signedinteger[Any]]: ...
    def __len__(self) -> int: ...

class LlamaConfig:
    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: str | Path, /, *args: Any, **kwargs: Any
    ) -> Self: ...

class AutoTokenizer:
    # Actually differs from the original implementation which doesn't have a return type hint
    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: str | Path | None, /, *args: Any, **kwargs: Any
    ) -> PreTrainedTokenizer: ...

class LlamaForCausalLM(PreTrainedModel): ...
class GPT2LMHeadModel(PreTrainedModel): ...
class AutoModelForCausalLM(PreTrainedModel): ...
