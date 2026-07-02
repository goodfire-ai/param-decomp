"""Build the in-memory `a x b` modular-arithmetic eval probe (the `ArithmeticCIGrid` metric).

The probe is the `[a_range] x [b_range]` grid of `"<a><symbol><b>="` prompts, tokenized
with the target's tokenizer — one prompt per row, all rows one shared token length, so the
`=` answer token sits at a fixed position and per-component CI / activation vectors
reshape into `a x b` heatmaps (row-major `(a, b)`). Unlike the streaming corpus there is
no packing and no offline artifact: the probe is a pure function of the metric spec + the
tokenizer, so every rank builds the identical grid at startup with no coordination.

For Llama-3.1 every 1-3 digit integer is a single token, so `"<a>+<b>="` is a constant
4 tokens (5 with BOS) and every sum 2..200 is a single token — the asserts below codify
that; a range/op that breaks either invariant fails fast rather than silently padding.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from param_decomp.arithmetic_eval import ArithmeticGrid

# Operation -> (display symbol, result fn). Only addition is exercised today; subtraction
# (negative results) and multiplication tokenize differently and must re-clear the
# single-token-answer assert before use.
OPERATIONS = {
    "add": ("+", lambda a, b: a + b),
    "sub": ("-", lambda a, b: a - b),
    "mul": ("*", lambda a, b: a * b),
}


class PromptEncoder(Protocol):
    """The slice of a HF tokenizer the probe needs (`PreTrainedTokenizer.encode`)."""

    def encode(self, text: str, /, *, add_special_tokens: bool) -> Sequence[int] | np.ndarray: ...


@dataclass(frozen=True)
class ArithmeticProbe:
    """`tokens[i * grid.n_b + j]` is the prompt for `a_values[i] <op> b_values[j] =`."""

    tokens: np.ndarray
    """`(n_prompts, T)` int32, row-major `(a, b)` order."""
    grid: ArithmeticGrid
    answer_position: int
    """The `=` token; its logits predict the (single-token) answer."""


def build_arithmetic_probe(
    operation: str,
    a_range: tuple[int, int],
    b_range: tuple[int, int],
    tokenizer: PromptEncoder,
) -> ArithmeticProbe:
    assert operation in OPERATIONS, f"operation must be one of {sorted(OPERATIONS)}"
    symbol, result_fn = OPERATIONS[operation]
    a_values = tuple(range(a_range[0], a_range[1] + 1))
    b_values = tuple(range(b_range[0], b_range[1] + 1))
    assert a_values and b_values, (a_range, b_range)

    rows: list[list[int]] = []
    seq_len: int | None = None
    for a in a_values:
        for b in b_values:
            prompt = f"{a}{symbol}{b}="
            ids = [int(t) for t in tokenizer.encode(prompt, add_special_tokens=True)]
            if seq_len is None:
                seq_len = len(ids)
            assert len(ids) == seq_len, (
                f"prompt {prompt!r} tokenizes to {len(ids)} tokens but expected {seq_len}; "
                f"all prompts must share one length (padding is disabled) — pick an operand "
                f"range whose operands are all single tokens"
            )
            answer = result_fn(a, b)
            answer_tokens = tokenizer.encode(str(answer), add_special_tokens=False)
            assert len(answer_tokens) == 1, (
                f"answer {answer} for {prompt!r} is {len(answer_tokens)} tokens, not 1; the "
                f"probe premise is a single answer token predicted at the `=` position"
            )
            rows.append(ids)
    assert seq_len is not None
    return ArithmeticProbe(
        tokens=np.asarray(rows, dtype=np.int32),
        grid=ArithmeticGrid(a_values=a_values, b_values=b_values, symbol=symbol),
        answer_position=seq_len - 1,
    )
