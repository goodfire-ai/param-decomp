"""Tokenizer for OthelloGPT.

Maps token indices 0-60 to board position strings.
Token 0 = [PAD], tokens 1-60 = the 60 playable squares in row-major order
(A1, A2, ..., H8, skipping center squares D4, D5, E4, E5).
"""

from pathlib import Path

from tokenizers import Tokenizer, models, pre_tokenizers

_CENTER_SQUARES = {27, 28, 35, 36}
_PLAYABLE = [i for i in range(64) if i not in _CENTER_SQUARES]
_ROWS = "ABCDEFGH"
_COLS = "12345678"

VOCAB: list[str] = ["[PAD]"] + [f"{_ROWS[sq // 8]}{_COLS[sq % 8]}" for sq in _PLAYABLE]
assert len(VOCAB) == 61


def token_to_square(token_id: int) -> str:
    return VOCAB[token_id]


def save_tokenizer(path: Path) -> None:
    """Save a HuggingFace-compatible tokenizer to disk."""
    vocab = {tok: i for i, tok in enumerate(VOCAB)}
    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[PAD]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()  # pyright: ignore[reportAttributeAccessIssue]
    path.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(path / "tokenizer.json"))
