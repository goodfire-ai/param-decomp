"""The tPD TARGET stream for an LM: a fixed prompt pool, tokenized once at
startup — deterministically on every rank, like the arithmetic probe. Every prompt must
tokenize to ONE shared length, so the target pass runs unpadded at the pool's own
natural geometry, independent of the broad stream's."""

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
from pydantic import Discriminator

from param_decomp.core.base_config import BaseConfig
from param_decomp.experiments.lm.arithmetic_probe import (
    PromptEncoder,
    build_arithmetic_prompt_grid,
)


class ArithmeticGridPromptsConfig(BaseConfig):
    """The `[a_range] x [b_range]` grid of `"<a><op><b>="` prompts — the same in-memory
    construction as the `ArithmeticCIGrid` eval probe, reused as a training pool.

    A pool draws prompt rows and never scores an answer position, so unlike the probe it
    carries NO single-token-answer premise — only the shared prompt length. Under a
    per-digit number tokenizer (the Qwen family) that means same-digit-count operands;
    single-digit-ANSWER ranges (e.g. add over `[1, 4] x [1, 5]`) are the working default
    for arithmetic-style pools, keeping the same grid usable as the eval probe too."""

    kind: Literal["arithmetic_grid"] = "arithmetic_grid"
    operation: Literal["add", "sub", "mul"]
    a_range: tuple[int, int]
    b_range: tuple[int, int]


class PromptsFileConfig(BaseConfig):
    """Ad-hoc escape hatch — a LOCATION, like `data: {kind: dir}`: one prompt per
    nonempty line, every line tokenizing to one shared length."""

    kind: Literal["prompts_file"] = "prompts_file"
    path: Path


LMPromptPoolConfig = Annotated[
    ArithmeticGridPromptsConfig | PromptsFileConfig, Discriminator("kind")
]


@dataclass(frozen=True)
class TargetPromptPool:
    """`tokens` is the whole pool, `(n_prompts, prompt_len)` int32 — unpadded, every row
    the one shared prompt length."""

    tokens: np.ndarray


def build_prompt_pool(
    config: ArithmeticGridPromptsConfig | PromptsFileConfig,
    tokenizer: PromptEncoder,
) -> TargetPromptPool:
    """Tokenize a fixed pool whose prompts all have the same length.

    The target stream runs at this length without padding. Training samples prompt
    rows; it does not require single-token answers or score an answer position."""
    match config:
        case ArithmeticGridPromptsConfig():
            tokens = build_arithmetic_prompt_grid(
                config.operation, config.a_range, config.b_range, tokenizer
            ).tokens
        case PromptsFileConfig():
            lines = [line for line in config.path.read_text().splitlines() if line.strip()]
            assert lines, f"no prompts in {config.path}"
            encoded = [
                np.asarray(tokenizer.encode(line, add_special_tokens=True), np.int32)
                for line in lines
            ]
            lengths = sorted({e.shape[0] for e in encoded})
            assert len(lengths) == 1, (
                f"prompts must tokenize to ONE shared length, got lengths {lengths} — the "
                "target stream is unpadded by construction"
            )
            tokens = np.stack(encoded)
    return TargetPromptPool(tokens=tokens)


def pool_batch(pool: TargetPromptPool, seed: int, step: int, global_batch: int) -> np.ndarray:
    """The step's global target batch: `global_batch` rows drawn uniformly (with
    replacement) from the pool — a pure function of `(seed, step)` (O(1) resume),
    identical on every rank; the caller slices its process share."""
    rng = np.random.default_rng(np.random.SeedSequence((seed, step)))
    indices = rng.integers(0, pool.tokens.shape[0], size=global_batch)
    return pool.tokens[indices]
