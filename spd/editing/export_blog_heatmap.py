"""Export write-vector editing KL heatmap data for the VPD blog post.

Generates real model data comparing SPD analytical editing vs LoRA baseline.
SPD: replace U row with scaled negated unembed (0 training examples).
LoRA: rank-1, all training examples, λ=10 KL regularization, 300 steps.

Usage:
    python -m spd.editing.export_blog_heatmap --out-dir /path/to/vpd-blog/data
"""

import json
import random
from collections.abc import Callable
from pathlib import Path

import torch
from torch import Tensor

from spd.app.backend.app_tokenizer import AppTokenizer
from spd.editing.component_trainer import u_replaced
from spd.editing.generate_pareto_plots import (
    get_examples,
    get_probs,
    kl_per_token,
    make_train_seqs,
    pad_train_seqs,
)
from spd.editing.lora_baseline import LoRATrainer
from spd.editing.utils import load_model
from spd.harvest.repo import HarvestRepo
from spd.harvest.schemas import ActivationExample

WANDB_PATH = "wandb:goodfire/spd/s-55ea3f9b"
RUN_ID = "s-55ea3f9b"
MODULE_NAME = "h.2.mlp.down_proj"
U_IDX = 2359
TARGET_TOKEN = 80  # "o"

ForwardFn = Callable[[Tensor], Tensor]


def export_diffs(
    forward_fn: ForwardFn,
    baselines: list[Tensor],
    token_tensors: list[Tensor],
    examples: list[ActivationExample],
    tok: AppTokenizer,
) -> list[dict[str, object]]:
    results = []
    with torch.no_grad():
        for tokens_t, probs_base, ex in zip(token_tensors, baselines, examples, strict=True):
            probs_edit = forward_fn(tokens_t.unsqueeze(0))[0].softmax(-1)
            kl = kl_per_token(probs_edit, probs_base)
            spans = tok.get_spans(tokens_t.tolist())
            fires = {i for i, f in enumerate(ex.firings) if f}

            tokens_out = []
            for t in range(len(spans)):
                before_idx = probs_base[t].topk(8).indices
                after_idx = probs_edit[t].topk(8).indices
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


def main(out_dir: Path) -> None:
    model, tok, _, _dl = load_model(WANDB_PATH, device="cuda", batch_size=40)
    harvest = HarvestRepo.open_most_recent(RUN_ID)
    assert harvest is not None

    examples = get_examples(harvest)
    random.seed(42)
    random.shuffle(examples)

    eval_examples = examples[:30]
    train_pool = examples[30:]
    eval_tokens = [torch.tensor(ex.token_ids, device="cuda") for ex in eval_examples]
    baselines = get_probs(model, eval_tokens)

    # SPD analytical
    lm_head = model.target_model.lm_head
    assert isinstance(lm_head, torch.nn.Linear)
    unembed = lm_head.weight[TARGET_TOKEN].detach().float()
    new_u = (-3.0 * unembed / unembed.norm()).to(torch.bfloat16)

    with u_replaced(model, MODULE_NAME, U_IDX, new_u) as spd_forward:
        spd_examples = export_diffs(spd_forward, baselines, eval_tokens, eval_examples, tok)
    print(f"SPD: {len(spd_examples)} examples")

    # LoRA
    train_seqs = make_train_seqs(train_pool)

    def forward_base(tokens: Tensor) -> Tensor:
        return model.target_model(tokens)[0]

    train_baselines = get_probs(forward_base, [t for t, _ in train_seqs])
    all_tokens, all_baselines, all_fire, all_pad = pad_train_seqs(train_seqs, train_baselines)
    n_train = all_tokens.shape[0]

    with LoRATrainer(model.target_model, MODULE_NAME, lr=1e-3, kl_weight=10.0) as lora:
        for step in range(300):
            idxs = torch.randint(n_train, (min(8, n_train),))
            lora.train_step(all_tokens[idxs], all_baselines[idxs], all_fire[idxs], all_pad[idxs])
            if step % 100 == 0:
                print(f"  LoRA step {step}")
        lora_examples = export_diffs(lora.forward, baselines, eval_tokens, eval_examples, tok)
    print(f"LoRA: {len(lora_examples)} examples")

    # Write
    out_dir.mkdir(exist_ok=True)
    for fname, data in [
        (
            "training-heatmap.json",
            {
                "component": f"{MODULE_NAME}:{U_IDX}",
                "method": "SPD analytical (α=3, 0 training examples)",
                "target_token": "o",
                "examples": spd_examples,
            },
        ),
        (
            "training-heatmap-lora.json",
            {
                "component": MODULE_NAME,
                "method": f"LoRA rank-1 (n={len(train_pool)}, λ=10, 300 steps)",
                "target_token": "o",
                "examples": lora_examples,
            },
        ),
    ]:
        (out_dir / fname).write_text(json.dumps(data, separators=(",", ":")))
    print(f"Written to {out_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    main(args.out_dir)
