"""Tokenizer wrapper that isolates HuggingFace tokenizer quirks from the rest of the app.

The core problem: `"".join(tokenizer.decode([t]) for t in ids)` != `tokenizer.decode(ids)`
because tokenizers encode word boundaries in family-specific ways (BPE's Ġ prefix,
WordPiece's ## prefix, SentencePiece's ▁ prefix, byte-level token splitting, etc.).

AppTokenizer provides two clean interfaces:
- get_spans(token_ids): per-token strings that concatenate to the full decoded text
- get_tok_display(token_id): single-token display string for vocab browsers / hover labels
"""

from typing import Self

from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

_CONTROL_CHAR_MAP = {
    "\t": "⇥",
    "\n": "↵",
    "\r": "⏎",
    "\x00": "␀",
}


def escape_for_display(s: str) -> str:
    """Escape control characters for human-readable display."""
    for char, replacement in _CONTROL_CHAR_MAP.items():
        s = s.replace(char, replacement)
    return s


class AppTokenizer:
    """Wraps a HuggingFace tokenizer. All decoding grossness lives here."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        self._tok = tokenizer
        self._is_fast = hasattr(tokenizer, "backend_tokenizer")

    @classmethod
    def from_pretrained(cls, tokenizer_name: str) -> Self:
        hf_tok = AutoTokenizer.from_pretrained(tokenizer_name)
        assert isinstance(hf_tok, PreTrainedTokenizerBase)
        return cls(hf_tok)

    @property
    def hf_tokenizer(self) -> PreTrainedTokenizerBase:
        """The underlying HuggingFace tokenizer, for APIs that require it directly."""
        return self._tok

    @property
    def vocab_size(self) -> int:
        size = self._tok.vocab_size
        assert isinstance(size, int)
        return size

    @property
    def eos_token_id(self) -> int:
        eos = self._tok.eos_token_id
        assert isinstance(eos, int)
        return eos

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text, add_special_tokens=False)

    def decode(self, token_ids: list[int]) -> str:
        return self._tok.decode(token_ids, skip_special_tokens=False)

    def get_spans(self, token_ids: list[int]) -> list[str]:
        """Decode token_ids into per-token display strings that concatenate to the full text.

        Uses offset_mapping (from the Rust tokenizer backend) when available, with dedup
        for overlapping byte-token spans. Falls back to per-token decode otherwise.
        """
        if not token_ids:
            return []

        if not self._is_fast:
            return self._fallback_spans(token_ids)

        text = self._tok.decode(token_ids, skip_special_tokens=False)
        re_encoded = self._tok(text, return_offsets_mapping=True, add_special_tokens=False)

        if re_encoded.input_ids != token_ids:
            return self._fallback_spans(token_ids)

        offsets: list[tuple[int, int]] = re_encoded.offset_mapping
        assert len(offsets) == len(token_ids)

        spans: list[str] = []
        prev_end = 0
        for start, end in offsets:
            if start >= prev_end:
                # Include any gap characters (spaces, etc.) as prefix of this span
                spans.append(text[prev_end:end])
                prev_end = end
            else:
                # Multi-byte char split across tokens: first token claimed the full char,
                # continuation byte-tokens get empty string
                spans.append("")

        assert "".join(spans) == text, f"span concat mismatch: {''.join(spans)!r} != {text!r}"
        return [escape_for_display(span) for span in spans]

    def get_tok_display(self, token_id: int) -> str:
        """Single token -> display string for vocab browsers and hover labels."""
        return escape_for_display(self._tok.decode([token_id], skip_special_tokens=False))

    def _fallback_spans(self, token_ids: list[int]) -> list[str]:
        """Incremental decode: each span = decode(:i+1) - decode(:i).

        O(n²) but correct for all tokenizer families (BPE, WordPiece, SentencePiece).
        """
        spans: list[str] = []
        prev = ""
        for i in range(len(token_ids)):
            current = self._tok.decode(token_ids[: i + 1], skip_special_tokens=False)
            spans.append(escape_for_display(current[len(prev) :]))
            prev = current
        return spans
