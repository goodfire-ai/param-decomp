"""Build arith_tokens.npy: the 10000 'a+b=' prompts (a, b in 1..100) as token ids.

Every number 1..100 is a single Llama-3.1 token, so each prompt is exactly
[BOS, a, +, b, =] and '=' is the final position. Row order a-outer/b-inner
(row i -> a = i//100 + 1, b = i%100 + 1) — same layout as p-1e7e8e36/p-594db290.
"""

from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

HERE = Path(__file__).parent


def main() -> None:
    tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
    rows = []
    for a in range(1, 101):
        for b in range(1, 101):
            ids = tok(f"{a}+{b}=", return_tensors="np")["input_ids"][0]
            assert ids.shape == (5,), (a, b, tok.convert_ids_to_tokens(ids.tolist()))
            rows.append(ids)
    tokens = np.stack(rows).astype(np.int32)
    assert tokens.shape == (10000, 5), tokens.shape
    np.save(HERE / "arith_tokens.npy", tokens)
    print(f"saved {HERE / 'arith_tokens.npy'} shape {tokens.shape}")
    print("first row:", tok.convert_ids_to_tokens(tokens[0].tolist()))
    print("last row:", tok.convert_ids_to_tokens(tokens[-1].tolist()))


if __name__ == "__main__":
    main()
