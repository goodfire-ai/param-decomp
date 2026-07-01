"""Generate + pre-tokenize a modular-arithmetic prompt GRID to a local eval artifact.

One-shot offline tool, sibling of `prestage_tokenized.py`: builds the `a x b` grid of
`"<a><symbol><b>="` prompts, tokenizes them with the target tokenizer, and writes a fixed
eval probe that the `arithmetic_eval` metric loads DIRECTLY (not through the streaming
`LMDataConfig` loader). Unlike the fineweb prestage this does NOT pack/concatenate — one
prompt per row, all rows one shared length, so the `=` answer token sits at a fixed
position. The artifact carries each prompt's `(a, b)` grid coordinates and ground-truth
`answer_id`, so per-component CI / activation vectors reshape into `a x b` heatmaps.

For Llama-3.1 every 1-3 digit integer is a single token, so `"<a>+<b>="` is a constant
4 tokens (5 with BOS) and every sum 2..200 is a single token — the asserts below codify
that; a range/op that breaks either invariant fails fast rather than silently padding.

Output: `<out_dir>/grid.parquet` (columns `a`, `b`, `answer_id`, `input_ids`) +
`<out_dir>/meta.json` (operation, symbol, seq_len, answer_position, grid shape, BOS flag,
tokenizer). Run:
`python -m param_decomp_lab.experiments.lm.prestage_arithmetic --out_dir <abs> [...]`
"""

import json
from pathlib import Path

import fire
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer

from param_decomp.log import logger

# Operation -> (display symbol, result fn). Only addition is exercised today; subtraction
# (negative results) and multiplication tokenize differently and must re-clear the
# single-token-answer assert before use.
OPERATIONS = {
    "add": ("+", lambda a, b: a + b),
    "sub": ("-", lambda a, b: a - b),
    "mul": ("*", lambda a, b: a * b),
}


def prestage(
    *,
    out_dir: str,
    operation: str = "add",
    a_min: int = 1,
    a_max: int = 100,
    b_min: int = 1,
    b_max: int = 100,
    tokenizer_name: str = "meta-llama/Llama-3.1-8B",
    add_bos: bool = True,
) -> None:
    """Write the `[a_min..a_max] x [b_min..b_max]` arithmetic grid as a fixed eval probe.

    Rows are emitted in row-major `(a, b)` order, but the artifact also stores `a`/`b`
    per row so the eval reshapes by coordinate rather than trusting row order.
    """
    assert operation in OPERATIONS, f"operation must be one of {sorted(OPERATIONS)}"
    symbol, result_fn = OPERATIONS[operation]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=True)

    a_values = list(range(a_min, a_max + 1))
    b_values = list(range(b_min, b_max + 1))

    rows_a: list[int] = []
    rows_b: list[int] = []
    answer_ids: list[int] = []
    input_ids: list[list[int]] = []
    seq_len: int | None = None
    for a in a_values:
        for b in b_values:
            prompt = f"{a}{symbol}{b}="
            ids = [int(t) for t in tokenizer.encode(prompt, add_special_tokens=add_bos)]
            if seq_len is None:
                seq_len = len(ids)
            assert len(ids) == seq_len, (
                f"prompt {prompt!r} tokenizes to {len(ids)} tokens but expected {seq_len}; "
                f"all prompts must share one length (padding is disabled) — pick an operand "
                f"range whose operands are all single tokens"
            )
            answer = result_fn(a, b)
            answer_tokens = [
                int(t) for t in tokenizer.encode(str(answer), add_special_tokens=False)
            ]
            assert len(answer_tokens) == 1, (
                f"answer {answer} for {prompt!r} is {len(answer_tokens)} tokens, not 1; the "
                f"accuracy/agreement metric reads a single answer token at the `=` position"
            )
            rows_a.append(a)
            rows_b.append(b)
            answer_ids.append(answer_tokens[0])
            input_ids.append(ids)
    assert seq_len is not None

    table = pa.table(
        {
            "a": pa.array(rows_a, pa.int32()),
            "b": pa.array(rows_b, pa.int32()),
            "answer_id": pa.array(answer_ids, pa.int32()),
            "input_ids": pa.array(
                np.asarray(input_ids, dtype=np.int32).tolist(), pa.list_(pa.int32())
            ),
        }
    )
    pq.write_table(table, out / "grid.parquet")

    meta = {
        "operation": operation,
        "symbol": symbol,
        "tokenizer_name": tokenizer_name,
        "add_bos": add_bos,
        "seq_len": seq_len,
        "answer_position": seq_len - 1,  # the `=` token; its logits predict the answer
        "n_a": len(a_values),
        "n_b": len(b_values),
        "a_values": a_values,
        "b_values": b_values,
        "n_prompts": len(input_ids),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    logger.info(
        f"wrote {len(input_ids)} prompts ({len(a_values)}x{len(b_values)} grid) "
        f"seq_len={seq_len} answer_pos={seq_len - 1} -> {out}"
    )


def cli() -> None:
    fire.Fire(prestage)


if __name__ == "__main__":
    cli()
