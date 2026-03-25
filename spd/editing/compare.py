"""Compare model outputs before and after component training."""

from dataclasses import dataclass

import torch
from jaxtyping import Float, Int
from torch import Tensor

from spd.app.backend.app_tokenizer import AppTokenizer
from spd.editing._editing import EditableModel, parse_component_key
from spd.editing.component_trainer import ComponentTrainer, TrainMode


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


def train_and_compare(
    em: EditableModel,
    tok: AppTokenizer,
    component_key: str,
    train_mode: TrainMode,
    train_seqs: list[tuple[Int[Tensor, " seq"], list[int]]],
    heldout_seqs: list[tuple[Int[Tensor, " seq"], list[int]]],
    target_token: int,
    lr: float = 1e-3,
    n_steps: int = 100,
    topk: int = 8,
) -> TrainResult:
    """Train a component and return per-token before/after comparisons on held-out.

    train_seqs/heldout_seqs: list of (token_ids, firing_positions). Train seqs should
    already have target positions mutated to target_token.
    """
    trainer = ComponentTrainer(em.model, targets={component_key: train_mode}, lr=lr)

    # Cache baseline logits + CI/activation on held-out
    baseline_logits: list[Float[Tensor, "seq vocab"]] = []
    with torch.no_grad():
        for tokens_t, _ in heldout_seqs:
            baseline_logits.append(trainer(tokens_t.unsqueeze(0))[0])

    # Train
    for _ in range(n_steps):
        for tokens_mut, positions in train_seqs:
            logits = trainer(tokens_mut.unsqueeze(0))
            pos_t = torch.tensor(positions, device=tokens_mut.device)
            loss = torch.nn.functional.cross_entropy(logits[0, positions], tokens_mut[pos_t + 1])
            trainer.step(loss)

    # Eval
    with torch.no_grad():
        train_probs = _eval_probs(trainer, train_seqs, target_token)
        heldout_probs = _eval_probs(trainer, heldout_seqs, target_token)
        diffs = _compute_diffs(trainer, em, tok, component_key, heldout_seqs, baseline_logits, topk)

    trainer.cleanup()

    diffs.sort(key=lambda d: -d.max_kl)
    return TrainResult(
        diffs=diffs,
        train_prob=sum(train_probs) / len(train_probs),
        heldout_prob=sum(heldout_probs) / len(heldout_probs),
        steps=n_steps,
    )


def _eval_probs(
    trainer: ComponentTrainer,
    seqs: list[tuple[Int[Tensor, " seq"], list[int]]],
    target_token: int,
) -> list[float]:
    probs = []
    for tokens_t, positions in seqs:
        logits = trainer(tokens_t.unsqueeze(0))
        for p in positions:
            probs.append(logits[0, p].softmax(-1)[target_token].item())
    return probs


def _compute_diffs(
    trainer: ComponentTrainer,
    em: EditableModel,
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
        probs_edit = trainer(tokens_t.unsqueeze(0))[0].softmax(-1)

        kl = (probs_edit * ((probs_edit + 1e-10).log() - (probs_base + 1e-10).log())).sum(-1)

        ci_map = em.get_ci(tokens_t)
        ci_vals = ci_map[module][:, cidx].cpu()
        act_vals = em.get_component_activations(tokens_t, component_key).cpu()

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
