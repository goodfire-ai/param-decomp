"""CPU tests for the lab-side arithmetic-probe glue: building the in-memory probe from the
grid spec (row-major order, the constant-length / single-token-answer fail-fasts) and
sharding it like a batch. The full in-loop run is exercised by the GPU smoke.
"""

import re

import jax
import numpy as np
import pytest
from jax.sharding import Mesh

from param_decomp_lab.experiments.lm.arithmetic_probe import build_arithmetic_probe
from param_decomp_lab.experiments.lm.run import _arithmetic_probe_global

BOS = 7
SYMBOL_IDS = {"+": 101, "-": 102, "*": 103, "=": 104}


class _StubTokenizer:
    """Llama-3.1-like: every integer below `split_from` is one token (`1000 + n`); larger
    ints split into digit tokens, to exercise the fail-fast asserts."""

    def __init__(self, split_from: int = 10**9):
        self.split_from = split_from

    def encode(self, text: str, add_special_tokens: bool) -> list[int]:
        ids = [BOS] if add_special_tokens else []
        for part in re.findall(r"\d+|[-+*=]", text):
            if part.isdigit():
                n = int(part)
                ids += [1000 + n] if n < self.split_from else [1000 + int(d) for d in part]
            else:
                ids.append(SYMBOL_IDS[part])
        return ids


def test_build_arithmetic_probe_grid_row_major_with_bos():
    probe = build_arithmetic_probe("add", (1, 3), (1, 4), _StubTokenizer())
    assert probe.tokens.shape == (12, 5) and probe.tokens.dtype == np.int32
    assert probe.answer_position == 4
    assert probe.grid.a_values == (1, 2, 3) and probe.grid.b_values == (1, 2, 3, 4)
    assert probe.grid.symbol == "+"
    # row-major (a, b): row 0 is "1+1=", row 1 is "1+2="; reshape lands (a=3, b=4) at [2, 3]
    assert probe.tokens[0].tolist() == [BOS, 1001, SYMBOL_IDS["+"], 1001, SYMBOL_IDS["="]]
    assert probe.tokens[1].tolist() == [BOS, 1001, SYMBOL_IDS["+"], 1002, SYMBOL_IDS["="]]
    grid_view = probe.grid.to_grid(probe.tokens)
    assert grid_view[2, 3].tolist() == [BOS, 1003, SYMBOL_IDS["+"], 1004, SYMBOL_IDS["="]]


def test_build_arithmetic_probe_operation_dispatch_and_rejects_unknown():
    probe = build_arithmetic_probe("mul", (2, 3), (2, 3), _StubTokenizer())
    assert probe.grid.symbol == "*"
    assert probe.tokens[0][2] == SYMBOL_IDS["*"]
    with pytest.raises(AssertionError, match="operation must be"):
        build_arithmetic_probe("div", (1, 2), (1, 2), _StubTokenizer())


def test_build_arithmetic_probe_rejects_multi_token_operand():
    # a=10 splits into two digit tokens -> prompt lengths diverge (sub keeps the answers
    # single-digit so the length assert, not the answer assert, is what fires)
    with pytest.raises(AssertionError, match="one length"):
        build_arithmetic_probe("sub", (9, 10), (1, 2), _StubTokenizer(split_from=10))


def test_build_arithmetic_probe_rejects_multi_token_answer():
    # operands single-token but 5+5=10 splits -> the answer premise breaks
    with pytest.raises(AssertionError, match="single answer token"):
        build_arithmetic_probe("add", (5, 5), (5, 6), _StubTokenizer(split_from=10))


def test_arithmetic_probe_global_preserves_grid():
    probe = build_arithmetic_probe("add", (1, 3), (1, 4), _StubTokenizer())
    devices = np.asarray(jax.devices()[:1]).reshape(1, 1)
    mesh = Mesh(devices, ("replicate", "fsdp"))
    sharded = _arithmetic_probe_global(probe.tokens, mesh, n_proc=1)
    # one device: no padding, rows preserved verbatim (the eval trims pad via n_prompts anyway)
    assert sharded.shape == probe.tokens.shape
    np.testing.assert_array_equal(np.asarray(sharded), probe.tokens)
