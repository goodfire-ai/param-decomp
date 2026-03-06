"""Prompt listing endpoints."""

import torch
from fastapi import APIRouter
from pydantic import BaseModel

from spd.app.backend.dependencies import DepLoadedRun, DepStateManager
from spd.app.backend.utils import log_errors

# =============================================================================
# Schemas
# =============================================================================


class PromptPreview(BaseModel):
    """Preview of a stored prompt for listing."""

    id: int
    token_ids: list[int]
    tokens: list[str]
    preview: str
    next_token_probs: list[float | None]  # Probability of next token (last is None)


PREVIEW_MAX_CHARS = 60


def _make_preview(spans: list[str]) -> str:
    text = "".join(spans)
    if len(text) <= PREVIEW_MAX_CHARS:
        return text
    return text[:PREVIEW_MAX_CHARS] + "..."


router = APIRouter(prefix="/api/prompts", tags=["prompts"])


def compute_next_token_probs(token_ids: list[int], loaded: DepLoadedRun) -> list[float | None]:
    """Compute P(next_token | prefix) for each position."""
    if len(token_ids) == 0:
        return []

    device = next(loaded.model.parameters()).device
    tokens_tensor = torch.tensor([token_ids], device=device)

    with torch.no_grad():
        logits = loaded.model(tokens_tensor)
        probs = torch.softmax(logits, dim=-1)

    result: list[float | None] = []
    for i in range(len(token_ids) - 1):
        next_token_id = token_ids[i + 1]
        prob = probs[0, i, next_token_id].item()
        result.append(prob)
    result.append(None)  # No next token for last position
    return result


@router.get("")
@log_errors
def list_prompts(manager: DepStateManager, loaded: DepLoadedRun) -> list[PromptPreview]:
    """Return list of all prompts for the loaded run with matching context length."""
    db = manager.db
    prompt_ids = db.get_all_prompt_ids(loaded.run.id, loaded.context_length)

    results: list[PromptPreview] = []
    for pid in prompt_ids:
        prompt = db.get_prompt(pid)
        assert prompt is not None, f"Prompt {pid} in index but not in DB"
        spans = loaded.tokenizer.get_spans(prompt.token_ids)
        next_token_probs = compute_next_token_probs(prompt.token_ids, loaded)
        results.append(
            PromptPreview(
                id=prompt.id,
                token_ids=prompt.token_ids,
                tokens=spans,
                preview=_make_preview(spans),
                next_token_probs=next_token_probs,
            )
        )
    return results


@router.post("/custom")
@log_errors
def add_custom_prompt(text: str, manager: DepStateManager, loaded: DepLoadedRun) -> PromptPreview:
    """Add a custom text prompt."""
    token_ids = loaded.tokenizer.encode(text)
    assert len(token_ids) > 0, "Text produced no tokens"

    # Truncate to context length
    token_ids = token_ids[: loaded.context_length]

    db = manager.db
    prompt_id = db.add_custom_prompt(loaded.run.id, token_ids, loaded.context_length)
    spans = loaded.tokenizer.get_spans(token_ids)
    next_token_probs = compute_next_token_probs(token_ids, loaded)

    return PromptPreview(
        id=prompt_id,
        token_ids=token_ids,
        tokens=spans,
        preview=_make_preview(spans),
        next_token_probs=next_token_probs,
    )
