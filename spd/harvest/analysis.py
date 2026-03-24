"""Query functions for analyzing harvest data.

These functions operate on storage classes from harvest/storage.py.
"""

import math
from dataclasses import dataclass
from typing import Literal, cast

import torch
from jaxtyping import Float
from torch import Tensor

from spd.app.backend.app_tokenizer import AppTokenizer
from spd.harvest.storage import CorrelationStorage, TokenStatsStorage

Metric = Literal["precision", "recall", "jaccard", "pmi"]


@dataclass
class CorrelatedComponent:
    """A component correlated with a query component, including raw counts."""

    component_key: str
    score: float
    count_i: int
    """Firing count of the query component"""
    count_j: int
    """Firing count of this component"""
    count_ij: int
    """Co-occurrence count"""
    count_total: int
    """Total tokens seen"""


@dataclass
class TokenPRLift:
    """Token precision, recall, lift, and PMI for a single component."""

    top_recall: list[tuple[str, float]]
    top_precision: list[tuple[str, float]]
    top_lift: list[tuple[str, float]]
    top_pmi: list[tuple[str, float]]
    bottom_pmi: list[tuple[str, float]] | None


def get_correlated_components(
    storage: CorrelationStorage,
    component_key: str,
    metric: Metric,
    top_k: int,
    largest: bool = True,
) -> list[CorrelatedComponent]:
    """Get top-k or bottom-k correlated components."""
    i = storage.key_to_idx[component_key]

    count_this = int(storage.count_i[i].item())
    if count_this == 0:
        return []

    count_others = storage.count_i
    cooccurrence_counts: Float[Tensor, " n_components"] = storage.count_ij[i].float()

    match metric:
        case "precision":
            scores = (cooccurrence_counts / count_this).nan_to_num(float("-inf"))
        case "recall":
            scores = (cooccurrence_counts / count_others).nan_to_num(float("-inf"))
        case "jaccard":
            intersection = cooccurrence_counts
            union = count_this + count_others - cooccurrence_counts
            scores = (intersection / union).nan_to_num(float("-inf"))
        case "pmi":
            p_this_that = cooccurrence_counts / storage.count_total
            p_this = count_this / storage.count_total
            p_that = count_others / storage.count_total
            lift = p_this_that / (p_this * p_that)
            scores = torch.log(lift).nan_to_num(float("-inf"))

    # Exclude self and inactive components
    scores[i] = float("-inf")
    scores[storage.count_i == 0] = float("-inf")
    scores[cooccurrence_counts == 0] = float("-inf")

    top_k_clamped = min(top_k, len(scores))
    top_values, top_indices = torch.topk(scores, top_k_clamped, largest=largest)

    output = []
    for idx, val in zip(top_indices.tolist(), top_values.tolist(), strict=True):
        if val == float("-inf"):
            continue
        assert math.isfinite(val), (
            f"Unexpected non-finite score {val} for {storage.component_keys[idx]}"
        )
        output.append(
            CorrelatedComponent(
                component_key=storage.component_keys[idx],
                score=val,
                count_i=count_this,
                count_j=int(storage.count_i[idx].item()),
                count_ij=int(cooccurrence_counts[idx].item()),
                count_total=storage.count_total,
            )
        )

    return output


def has_component(storage: CorrelationStorage, component_key: str) -> bool:
    """Check if a component exists in the storage."""
    return component_key in storage.key_to_idx


def get_input_token_stats(
    storage: TokenStatsStorage,
    component_key: str,
    tok: AppTokenizer,
    top_k: int,
) -> TokenPRLift | None:
    """Compute P/R/lift/PMI for input tokens."""
    idx = storage.key_to_idx[component_key]

    return _compute_token_stats(
        counts=storage.input_counts[idx],
        totals=storage.input_totals,
        n_tokens=storage.n_tokens,
        firing_count=storage.firing_counts[idx].item(),
        tok=tok,
        top_k=top_k,
    )


def get_output_token_stats(
    storage: TokenStatsStorage,
    component_key: str,
    tok: AppTokenizer,
    top_k: int,
    pmi_min_count: float = 0.0,
) -> TokenPRLift | None:
    """Compute P/R/lift/PMI for output tokens."""
    idx = storage.key_to_idx[component_key]

    return _compute_token_stats(
        counts=storage.output_counts[idx],
        totals=storage.output_totals,
        n_tokens=storage.n_tokens,
        firing_count=storage.firing_counts[idx].item(),
        top_k=top_k,
        tok=tok,
        pmi_min_count=pmi_min_count,
    )


def _compute_token_stats(
    counts: Float[Tensor, " vocab"],
    totals: Float[Tensor, " vocab"],
    n_tokens: int,
    firing_count: float,
    tok: AppTokenizer,
    top_k: int,
    pmi_min_count: float = 0.0,
) -> TokenPRLift | None:
    """Compute P/R/lift/PMI from count tensors."""
    if firing_count == 0:
        return None

    valid_mask = (counts > 0) & (totals > 0)
    if not valid_mask.any():
        return None
    pmi_valid_mask = valid_mask & (counts >= pmi_min_count)

    recall = counts / firing_count
    precision = torch.where(totals > 0, counts / totals, torch.zeros_like(counts))
    base_rate = firing_count / n_tokens
    lift = precision / base_rate if base_rate > 0 else torch.zeros_like(precision)

    pmi = torch.log(counts * n_tokens / (firing_count * totals))
    pmi = torch.where(pmi_valid_mask, pmi, torch.full_like(pmi, float("-inf")))

    def get_top_k(
        values: Tensor,
        k: int,
        largest: bool = True,
        mask: Tensor | None = None,
    ) -> list[tuple[str, float]]:
        active_mask = valid_mask if mask is None else mask
        n_active = int(active_mask.sum().item())
        if n_active == 0 or k == 0:
            return []
        masked = torch.where(
            active_mask,
            values,
            torch.full_like(values, float("-inf") if largest else float("inf")),
        )
        top_vals, top_idx = torch.topk(
            masked,
            min(k, n_active),
            largest=largest,
        )

        result: list[tuple[str, float]] = []

        for idx, val in zip(
            cast(list[int], top_idx.tolist()), cast(list[float], top_vals.tolist()), strict=True
        ):
            if val == float("-inf"):
                continue
            assert math.isfinite(val), f"Unexpected non-finite score {val} for token {idx}"
            result.append((tok.get_tok_display(idx), round(val, 3 if abs(val) < 10 else 2)))
        return result

    return TokenPRLift(
        top_recall=get_top_k(recall, top_k),
        top_precision=get_top_k(precision, top_k),
        top_lift=get_top_k(lift, top_k),
        top_pmi=get_top_k(pmi, top_k, largest=True, mask=pmi_valid_mask),
        bottom_pmi=get_top_k(pmi, top_k, largest=False, mask=pmi_valid_mask),
    )
