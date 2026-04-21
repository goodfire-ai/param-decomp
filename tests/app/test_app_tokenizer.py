"""Tests for AppTokenizer span reconstruction and display logic."""

import pytest
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from spd.app.backend.app_tokenizer import AppTokenizer

# Test strings covering various tokenization edge cases
BASIC_STRINGS = [
    "Hello, world!",
    "The quick brown fox jumps over the lazy dog.",
    "It's a beautiful day.",
    "price is $19.99",
    "foo    bar",  # multiple spaces
    "line1\nline2",  # newline
]

UNICODE_STRINGS = [
    "café résumé naïve",
    "日本語テスト",
]


@pytest.fixture(scope="module")
def gpt2_tokenizer() -> AppTokenizer:
    tok = AutoTokenizer.from_pretrained("openai-community/gpt2")
    assert isinstance(tok, PreTrainedTokenizerBase)
    return AppTokenizer(tok)


class TestGetSpans:
    """Test that get_spans produces strings that concatenate to the full decoded text."""

    def test_empty(self, gpt2_tokenizer: AppTokenizer) -> None:
        assert gpt2_tokenizer.get_spans([]) == []

    @pytest.mark.parametrize("text", BASIC_STRINGS)
    def test_round_trip_basic(self, gpt2_tokenizer: AppTokenizer, text: str) -> None:
        token_ids = gpt2_tokenizer.encode(text)
        spans = gpt2_tokenizer.get_spans(token_ids)
        assert len(spans) == len(token_ids)
        from spd.app.backend.app_tokenizer import escape_for_display

        assert "".join(spans) == escape_for_display(gpt2_tokenizer.decode(token_ids))

    @pytest.mark.parametrize("text", UNICODE_STRINGS)
    def test_round_trip_unicode(self, gpt2_tokenizer: AppTokenizer, text: str) -> None:
        token_ids = gpt2_tokenizer.encode(text)
        spans = gpt2_tokenizer.get_spans(token_ids)
        assert len(spans) == len(token_ids)
        # For unicode, some spans may be empty (multi-byte split), but concat must match
        from spd.app.backend.app_tokenizer import escape_for_display

        assert "".join(spans) == escape_for_display(gpt2_tokenizer.decode(token_ids))

    def test_single_token(self, gpt2_tokenizer: AppTokenizer) -> None:
        token_ids = gpt2_tokenizer.encode("hi")
        assert len(token_ids) == 1
        spans = gpt2_tokenizer.get_spans(token_ids)
        assert spans == [gpt2_tokenizer.decode(token_ids)]


class TestGetTokDisplay:
    """Test single-token display strings."""

    def test_known_tokens(self, gpt2_tokenizer: AppTokenizer) -> None:
        # Token 0 is "!" in GPT-2
        display = gpt2_tokenizer.get_tok_display(0)
        assert isinstance(display, str)
        assert len(display) > 0

    def test_space_token(self, gpt2_tokenizer: AppTokenizer) -> None:
        # " the" is a common GPT-2 token
        token_ids = gpt2_tokenizer.encode(" the")
        assert len(token_ids) == 1
        display = gpt2_tokenizer.get_tok_display(token_ids[0])
        assert "the" in display


class TestEncodeDecode:
    """Test encode/decode round-trip."""

    def test_encode_decode(self, gpt2_tokenizer: AppTokenizer) -> None:
        text = "Hello, world!"
        token_ids = gpt2_tokenizer.encode(text)
        decoded = gpt2_tokenizer.decode(token_ids)
        assert decoded == text

    def test_vocab_size(self, gpt2_tokenizer: AppTokenizer) -> None:
        assert gpt2_tokenizer.vocab_size == 50257
