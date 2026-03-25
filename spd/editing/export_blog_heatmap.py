"""Export write-vector editing KL heatmap data for the VPD blog post.

Generates real model data comparing SPD analytical editing vs LoRA baseline.
SPD: replace U row with scaled negated unembed (0 training examples).
LoRA: rank-1, 1000+ training examples, λ=10 KL regularization.

Usage:
    python -m spd.editing.export_blog_heatmap \\
        --out-dir /path/to/vpd-blog/data
"""

import json
import random
from collections.abc import Callable
from pathlib import Path

import torch
from torch import Tensor

from spd.app.backend.app_tokenizer import AppTokenizer
from spd.data import train_loader_and_tokenizer
from spd.editing.component_trainer import write_edit
from spd.editing.lora_baseline import LoRATrainer
from spd.editing.utils import load_model
from spd.harvest.repo import HarvestRepo
from spd.harvest.schemas import ActivationExample

WANDB_PATH = "wandb:goodfire/spd/s-55ea3f9b"
RUN_ID = "s-55ea3f9b"
COMP_KEY = "h.2.mlp.down_proj:2359"
TARGET_TOKEN = 80  # "o"
LAYER_PATH = "h.2.mlp.down_proj"

ForwardFn = Callable[[Tensor], Tensor]


def fire_positions(ex: ActivationExample) -> list[int]:
    return [i for i, f in enumerate(ex.firings) if f and i + 1 < len(ex.token_ids)]


def get_examples(harvest: HarvestRepo) -> list[ActivationExample]:
    comp = harvest.get_component(COMP_KEY)
    assert comp is not None
    return [ex for ex in comp.activation_examples if fire_positions(ex)]


def export_diffs(
    forward_fn: ForwardFn,
    baseline_probs: dict[int, Tensor],
    token_tensors: list[Tensor],
    examples: list[ActivationExample],
    tok: AppTokenizer,
    topk: int = 8,
) -> list[dict[str, object]]:
    results = []
    with torch.no_grad():
        for tokens_t, ex in zip(token_tensors, examples, strict=True):
            probs_edit = forward_fn(tokens_t.unsqueeze(0))[0].softmax(-1)
            probs_base = baseline_probs[id(tokens_t)]
            kl = (probs_edit * ((probs_edit + 1e-10).log() - (probs_base + 1e-10).log())).sum(-1)
            spans = tok.get_spans(tokens_t.tolist())
            fires = {i for i, f in enumerate(ex.firings) if f}

            tokens_out = []
            for t in range(len(spans)):
                before_idx = probs_base[t].topk(topk).indices
                after_idx = probs_edit[t].topk(topk).indices
                tokens_out.append(
                    {
                        "span": spans[t],
                        "kl": round(kl[t].item(), 6),
                        "fires": t in fires,
                        "topk_before": [
                            [tok.get_tok_display(int(j)), round(probs_base[t, j].item(), 4)]
                            for j in before_idx
                        ],
                        "topk_after": [
                            [tok.get_tok_display(int(j)), round(probs_edit[t, j].item(), 4)]
                            for j in after_idx
                        ],
                    }
                )
            results.append({"tokens": tokens_out})
    return results


def cache_baselines(forward_fn: ForwardFn, token_tensors: list[Tensor]) -> dict[int, Tensor]:
    baselines = {}
    with torch.no_grad():
        for t in token_tensors:
            baselines[id(t)] = forward_fn(t.unsqueeze(0))[0].softmax(-1)
    return baselines


def main(out_dir: Path) -> None:
    model, tok, config = load_model(WANDB_PATH)
    harvest = HarvestRepo.open_most_recent(RUN_ID)
    assert harvest is not None

    examples = get_examples(harvest)
    random.seed(42)
    random.shuffle(examples)

    # Eval: first 30 examples, train: rest
    eval_examples = examples[:30]
    eval_tokens = [torch.tensor(ex.token_ids, device="cuda") for ex in eval_examples]

    # --- SPD analytical: U = -3 * unembed('o') / |unembed('o')| ---
    lm_head = model.target_model.lm_head
    assert isinstance(lm_head, torch.nn.Linear)
    unembed = lm_head.weight[TARGET_TOKEN].detach().float()
    u_delta = -3.0 * unembed / unembed.norm()

    def base_forward(tokens: Tensor) -> Tensor:
        return model._extract_output(model.target_model(tokens))

    true_baselines = cache_baselines(base_forward, eval_tokens)

    with write_edit(model, COMP_KEY, u_delta) as spd_forward:
        spd_examples = export_diffs(spd_forward, true_baselines, eval_tokens, eval_examples, tok)

    print(f"SPD: {len(spd_examples)} examples")

    # --- LoRA: n=all, λ=10, 300 steps ---
    train_pool = examples[30:]
    train_seqs = []
    for ex in train_pool:
        t = torch.tensor(ex.token_ids, device="cuda")
        positions = fire_positions(ex)
        for pos in positions:
            t[pos + 1] = TARGET_TOKEN
        train_seqs.append((t, positions))

    dl, _ = train_loader_and_tokenizer(config, batch_size=20)
    reg_seqs = [row.cuda() for row in next(iter(dl))["input_ids"]]

    lora = LoRATrainer(model.target_model, LAYER_PATH, reg_seqs, lr=1e-3)

    for step in range(300):
        lora.train_step(train_seqs, kl_weight=10.0)
        if step % 100 == 0:
            print(f"  LoRA step {step}")

    lora_examples = export_diffs(lora.forward, true_baselines, eval_tokens, eval_examples, tok)
    lora.cleanup()

    print(f"LoRA: {len(lora_examples)} examples")

    # Write
    out_dir.mkdir(exist_ok=True)
    (out_dir / "training-heatmap.json").write_text(
        json.dumps(
            {
                "component": COMP_KEY,
                "label": "Punctuation marks starting an emoticon",
                "target_token": "o",
                "examples": spd_examples,
            },
            separators=(",", ":"),
        )
    )
    (out_dir / "training-heatmap-lora.json").write_text(
        json.dumps(
            {
                "component": f"{LAYER_PATH} (rank-1 LoRA)",
                "label": "Punctuation marks starting an emoticon",
                "target_token": "o",
                "examples": lora_examples,
            },
            separators=(",", ":"),
        )
    )
    print(f"Written to {out_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    main(args.out_dir)
