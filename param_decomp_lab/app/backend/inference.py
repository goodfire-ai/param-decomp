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

    output_probs = jax_run.forward(jnp.asarray([token_ids])).output_probs[0]
    next_tokens = jnp.asarray(token_ids[1:])
    next_probs = np.asarray(output_probs[jnp.arange(len(token_ids) - 1), next_tokens])
    result: list[float | None] = [float(p) for p in next_probs]
    result.append(None)
    return result
