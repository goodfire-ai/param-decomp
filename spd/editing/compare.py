"""Compare model outputs before and after component write-vector editing."""

from collections.abc import Callable
from dataclasses import dataclass

import torch
from jaxtyping import Float, Int
from torch import Tensor

from spd.app.backend.app_tokenizer import AppTokenizer
from spd.editing._editing import get_ci, get_component_activations, parse_component_key
from spd.editing.component_trainer import train_write_delta, write_edit
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


@dataclass
class TrainResult:
    diffs: list[ExampleDiff]
    train_prob: float
    heldout_prob: float
    steps: int
    u_delta: Float[Tensor, " d_out"]


def train_and_compare(
    model: ComponentModel,
    tok: AppTokenizer,
    component_key: str,
    train_seqs: list[tuple[Int[Tensor, " seq"], list[int]]],
    heldout_seqs: list[tuple[Int[Tensor, " seq"], list[int]]],
    target_token: int,
    lr: float = 1e-3,
    n_steps: int = 100,
    topk: int = 8,
) -> TrainResult:
    """Train a component's write vector and return per-token before/after comparisons.

    train_seqs should already have target positions mutated to target_token.
    """
    forward_base = lambda tokens: model._extract_output(model.target_model(tokens))

    baseline_logits: list[Float[Tensor, "seq vocab"]] = []
    with torch.no_grad():
        for tokens_t, _ in heldout_seqs:
            baseline_logits.append(forward_base(tokens_t.unsqueeze(0))[0])

    u_delta = train_write_delta(model, component_key, train_seqs, lr=lr, n_steps=n_steps)

    with write_edit(model, component_key, u_delta) as forward_fn, torch.no_grad():
        train_probs = _eval_probs(forward_fn, train_seqs, target_token)
        heldout_probs = _eval_probs(forward_fn, heldout_seqs, target_token)
        diffs = _compute_diffs(
            forward_fn, model, tok, component_key, heldout_seqs, baseline_logits, topk
        )

    diffs.sort(key=lambda d: -d.max_kl)
    return TrainResult(
        diffs=diffs,
        train_prob=sum(train_probs) / len(train_probs),
        heldout_prob=sum(heldout_probs) / len(heldout_probs),
        steps=n_steps,
        u_delta=u_delta,
    )


def _eval_probs(
    forward_fn: Callable[[Tensor], Tensor],
    seqs: list[tuple[Int[Tensor, " seq"], list[int]]],
    target_token: int,
) -> list[float]:
    probs = []
    for tokens_t, positions in seqs:
        logits = forward_fn(tokens_t.unsqueeze(0))
        for p in positions:
            probs.append(logits[0, p].softmax(-1)[target_token].item())
    return probs


def _compute_diffs(
    forward_fn: Callable[[Tensor], Tensor],
    model: ComponentModel,
    tok: AppTokenizer,
    component_key: str,
    heldout_seqs: list[tuple[Int[Tensor, " seq"], list[int]]],
    baseline_logits: list[Float[Tensor, "seq vocab"]],
    topk: int,
) -> list[ExampleDiff]:
    module, cidx = parse_component_key(component_key)
    results = []

    for i, (tokens_t, positions) in enumerate(heldout_seqs):
        probs_base = baseline_logits[i].softmax(-1)
        probs_edit = forward_fn(tokens_t.unsqueeze(0))[0].softmax(-1)

        kl = (probs_edit * ((probs_edit + 1e-10).log() - (probs_base + 1e-10).log())).sum(-1)

        ci_map = get_ci(model, tokens_t)
        ci_vals = ci_map[module][:, cidx].cpu()
        act_vals = get_component_activations(model, tokens_t, component_key).cpu()

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
