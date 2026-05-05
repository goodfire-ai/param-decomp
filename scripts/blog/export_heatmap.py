"""Export editing KL heatmap data for the VPD blog post.

Compares PD analytical editing (0 training examples) vs LoRA baseline
(rank-1, all training examples, lambda=10 KL regularization, 300 steps).

Run from ~/param-decomp:
  uv run python scripts/blog/export_heatmap.py --out-dir ../vpd-blog-replit/data
"""

import argparse
import json
import random
from collections.abc import Callable
from pathlib import Path

import torch
from param_decomp.editing.component_trainer import u_replaced
from param_decomp.editing.generate_pareto_plots import (
    get_examples,
    get_probs,
    kl_per_token,
    make_train_seqs,
    pad_train_seqs,
)
from param_decomp.editing.lora_baseline import LoRATrainer
from param_decomp.editing.utils import load_model
from torch import Tensor

from param_decomp.app.backend.app_tokenizer import AppTokenizer
from param_decomp.harvest.repo import HarvestRepo
from param_decomp.harvest.schemas import ActivationExample
from scripts.blog.constants import (
    HEATMAP_MODULE,
    HEATMAP_TARGET_TOKEN,
    HEATMAP_U_IDX,
    RUN_ID,
    WANDB_PATH,
)

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory")
    args = parser.parse_args()
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

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

    # PD analytical edit
    lm_head = model.target_model.lm_head
    assert isinstance(lm_head, torch.nn.Linear)
    unembed = lm_head.weight[HEATMAP_TARGET_TOKEN].detach().float()
    new_u = (-3.0 * unembed / unembed.norm()).to(torch.bfloat16)

    with u_replaced(model, HEATMAP_MODULE, HEATMAP_U_IDX, new_u) as pd_forward:
        pd_examples = export_diffs(pd_forward, baselines, eval_tokens, eval_examples, tok)
    print(f"PD: {len(pd_examples)} examples")

    # LoRA baseline
    train_seqs = make_train_seqs(train_pool)

    def forward_base(tokens: Tensor) -> Tensor:
        return model.target_model(tokens)[0]

    train_baselines = get_probs(forward_base, [t for t, _ in train_seqs])
    all_tokens, all_baselines, all_fire, all_pad = pad_train_seqs(train_seqs, train_baselines)
    n_train = all_tokens.shape[0]

    with LoRATrainer(model.target_model, HEATMAP_MODULE, lr=1e-3, kl_weight=10.0) as lora:
        for step in range(300):
            idxs = torch.randint(n_train, (min(8, n_train),))
            lora.train_step(all_tokens[idxs], all_baselines[idxs], all_fire[idxs], all_pad[idxs])
            if step % 100 == 0:
                print(f"  LoRA step {step}")
        lora_examples = export_diffs(lora.forward, baselines, eval_tokens, eval_examples, tok)
    print(f"LoRA: {len(lora_examples)} examples")

    for fname, data in [
        (
            "editing-kl-heatmap.json",
            {
                "component": f"{HEATMAP_MODULE}:{HEATMAP_U_IDX}",
                "method": "PD analytical (alpha=3, 0 training examples)",
                "target_token": "o",
                "examples": pd_examples,
            },
        ),
        (
            "editing-kl-heatmap-lora.json",
            {
                "component": HEATMAP_MODULE,
                "method": f"LoRA rank-1 (n={len(train_pool)}, lambda=10, 300 steps)",
                "target_token": "o",
                "examples": lora_examples,
            },
        ),
    ]:
        (out_dir / fname).write_text(json.dumps(data, separators=(",", ":")))
    print(f"Written to {out_dir}")


if __name__ == "__main__":
    main()
