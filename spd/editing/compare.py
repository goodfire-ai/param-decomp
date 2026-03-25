"""Compare model outputs before and after component write-vector editing."""

from collections.abc import Callable
from dataclasses import dataclass

import torch
from jaxtyping import Float, Int
from torch import Tensor

from spd.app.backend.app_tokenizer import AppTokenizer
from spd.editing.utils import parse_component_key
from spd.models.component_model import ComponentModel


@dataclass
class TokenDiff:
    span: str
    kl: float
    ci: float
    activation: float
    fires: bool
    topk_before: list[tuple[str, float]]
    topk_after: list[tuple[str, float]]
    top_increases: list[tuple[str, float, float]]
    top_decreases: list[tuple[str, float, float]]


@dataclass
class ExampleDiff:
    tokens: list[TokenDiff]
    max_kl: float


def compute_diffs(
    forward_fn: Callable[[Tensor], Tensor],
    model: ComponentModel,
    tok: AppTokenizer,
    component_key: str,
    seqs: list[tuple[Int[Tensor, " seq"], list[int]]],
    baseline_logits: list[Float[Tensor, "seq vocab"]],
    topk: int = 8,
) -> list[ExampleDiff]:
    """Per-token before/after diffs for a set of sequences."""
    module, cidx = parse_component_key(component_key)
    results = []

    for i, (tokens_t, positions) in enumerate(seqs):
        probs_base = baseline_logits[i].softmax(-1)
        probs_edit = forward_fn(tokens_t.unsqueeze(0))[0].softmax(-1)
        kl = (probs_edit * ((probs_edit + 1e-10).log() - (probs_base + 1e-10).log())).sum(-1)

        with torch.no_grad():
            out = model(tokens_t.unsqueeze(0), cache_type="input")
            ci = model.calc_causal_importances(
                pre_weight_acts=out.cache, sampling="continuous", detach_inputs=False
            )
        ci_vals = ci.lower_leaky[module].squeeze(0)[:, cidx].cpu()

        pre_weight_acts = out.cache[module]  # [1, seq, d_in]
        act_vals = (pre_weight_acts @ model.components[module].V[:, cidx]).squeeze(0).cpu()

        spans = tok.get_spans(tokens_t.tolist())
        firing_set = set(positions)

        token_diffs = []
        for t in range(len(spans)):
            diff = probs_edit[t] - probs_base[t]
            inc_idx = diff.topk(5).indices
            dec_idx = (-diff).topk(5).indices
            before_idx = probs_base[t].topk(topk).indices
            after_idx = probs_edit[t].topk(topk).indices

            token_diffs.append(
                TokenDiff(
                    span=spans[t],
                    kl=kl[t].item(),
                    ci=ci_vals[t].item(),
                    activation=act_vals[t].item(),
                    fires=t in firing_set,
                    topk_before=[
                        (tok.get_tok_display(int(j.item())), probs_base[t, j].item())
                        for j in before_idx
                    ],
                    topk_after=[
                        (tok.get_tok_display(int(j.item())), probs_edit[t, j].item())
                        for j in after_idx
                    ],
                    top_increases=[
                        (
                            tok.get_tok_display(int(j.item())),
                            probs_base[t, j].item(),
                            probs_edit[t, j].item(),
                        )
                        for j in inc_idx
                    ],
                    top_decreases=[
                        (
                            tok.get_tok_display(int(j.item())),
                            probs_base[t, j].item(),
                            probs_edit[t, j].item(),
                        )
                        for j in dec_idx
                    ],
                )
            )

        results.append(ExampleDiff(tokens=token_diffs, max_kl=kl.max().item()))

    return results
