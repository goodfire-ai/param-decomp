"""Fixed-prompt target stream for targeted PD (SPEC S34): a small newline-separated prompt
pool tokenized once at build time, sampled (with replacement) into per-step token batches.
The non-target stream stays the normal parquet path (`experiments.lm` loaders)."""

from collections.abc import Callable
from pathlib import Path

import jax
import numpy as np
from jax import random
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from transformers import AutoTokenizer


def load_prompt_tokens(
    prompts_file: str, tokenizer_name: str, max_seq_len: int, add_special_tokens: bool
) -> tuple[np.ndarray, int]:
    """Tokenize each non-empty line of `prompts_file` into a fixed `[n_prompts, max_seq_len]`
    int32 array, returning `(tokens, recon_positions)` where `recon_positions` is the real
    (pre-pad) prompt length. tPD targets are short constant-length prompts (e.g. "import numpy
    as"); they are END-padded (token 0) to `max_seq_len` so the forward clears the cuDNN
    flash-attn min-seq. Causal attention makes trailing pad invisible to the real positions,
    so recon is scored on only the first `recon_positions` positions (SPEC S38). All prompts
    must share one real length (so a single `recon_positions` applies to the batch).

    `add_special_tokens` must match the eval probe's tokenization: the ArithmeticCIGrid probe
    encodes with add_special_tokens=True (Llama BOS included), so the target prompts must too,
    else the target forward and the probe disagree on the leading BOS."""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=True)
    lines = [ln for ln in Path(prompts_file).read_text().splitlines() if ln.strip()]
    assert lines, f"no prompts in {prompts_file}"
    tokenized = [tokenizer.encode(ln, add_special_tokens=add_special_tokens) for ln in lines]
    lengths = {len(ids) for ids in tokenized}
    assert len(lengths) == 1, (
        f"all prompts must share one length (for a single recon_positions), got {lengths}"
    )
    recon_positions = lengths.pop()
    assert recon_positions <= max_seq_len, (
        f"prompt length {recon_positions} exceeds max_seq_len {max_seq_len}"
    )
    rows = [ids + [0] * (max_seq_len - recon_positions) for ids in tokenized]
    return np.asarray(rows, dtype=np.int32), recon_positions


def make_prompt_sample_batch(
    tokens: np.ndarray, global_batch: int, mesh: Mesh, seed: int
) -> Callable[[int], jax.Array]:
    """Per-step target batch: sample `global_batch` prompts (with replacement) from the fixed
    pool and shard over the mesh — a pure function of `step`, like the parquet `sample_batch`."""
    pool = jax.numpy.asarray(tokens)
    n_prompts = int(tokens.shape[0])
    base_key = random.PRNGKey(seed)

    def sample_batch(step: int) -> jax.Array:
        idx = random.randint(random.fold_in(base_key, step), (global_batch,), 0, n_prompts)
        batch = pool[idx]
        return jax.lax.with_sharding_constraint(
            batch, NamedSharding(mesh, P(("replicate", "fsdp")))
        )

    return sample_batch
