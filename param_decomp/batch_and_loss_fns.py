"""Protocols for the callbacks `optimize()` invokes once per batch.

The lab ships concrete implementations in `param_decomp_lab.batch_and_loss_fns`.
"""

from typing import Any, Protocol

import torch
from jaxtyping import Float
from torch import Tensor, nn


class RunBatch(Protocol):
    """Callable that runs one batch through `model` and returns its output.

    The output type is experiment-defined (`Any`) — typically a tensor of logits, but
    may be a dataclass / dict carrying additional fields (attention masks, hidden
    states, labels) that the experiment's `ReconstructionLoss` consumes. The same
    `RunBatch` is invoked on both the frozen target and the decomposed model, so the
    two `output` values it produces share a structure.
    """

    def __call__(self, model: nn.Module, batch: Any) -> Any: ...


class ReconstructionLoss(Protocol):
    """Compare a decomposed-model `output` against the frozen-target `target_output`.

    Both are whatever the experiment's `RunBatch` returns. The return pair
    `(sum, n_elements)` is the unreduced sum of per-element losses and the count it
    summed over (or sum-of-weights for weighted/masked losses); callers reduce
    `sum / n_elements` to a mean as needed.

    Per-batch context the loss needs (padding masks, MLM-masked positions,
    per-channel weights, labels) rides on the `output` / `target_output` structure
    — experiments are responsible for packaging it inside `RunBatch`. Static aux
    state (e.g. a k-mer→nucleotide lookup table) lives in a closure / partial /
    `__call__`-bearing class — the Protocol stays minimal.
    """

    def __call__(
        self,
        output: Any,
        target_output: Any,
    ) -> tuple[Float[Tensor, ""], int]: ...


def move_batch_to_device(batch: Any, device: str | torch.device) -> Any:
    """Recursively move every `Tensor` in a (possibly nested) `batch` to `device`.

    Supports tensors, tuples, and dicts; passes other types through unchanged.
    """
    if isinstance(batch, Tensor):
        return batch.to(device)
    if isinstance(batch, tuple):
        return tuple(move_batch_to_device(x, device) for x in batch)
    if isinstance(batch, dict):
        return {k: move_batch_to_device(v, device) for k, v in batch.items()}
    return batch
