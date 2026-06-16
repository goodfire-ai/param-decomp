"""Torch-free next-token probabilities from a loaded JAX run.

`LoadedJaxRun.forward(...).output_probs` is the softmax of the clean (frozen-target)
logits — the same quantity the app used to compute with `torch.softmax(model(tokens))`.
"""

import jax.numpy as jnp
import numpy as np
from jax_single_pool.load_run import LoadedJaxRun


def next_token_probs(jax_run: LoadedJaxRun, token_ids: list[int]) -> list[float | None]:
    """P(token[i+1] | token[:i+1]) for each position; None for the last (no next token)."""
    if len(token_ids) == 0:
        return []

    output_probs = np.asarray(jax_run.forward(jnp.asarray([token_ids])).output_probs[0])
    result: list[float | None] = [
        float(output_probs[i, token_ids[i + 1]]) for i in range(len(token_ids) - 1)
    ]
    result.append(None)
    return result
