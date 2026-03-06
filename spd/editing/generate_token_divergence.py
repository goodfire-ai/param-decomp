"""Generate per-token divergence data for the token divergence visualisation.

Runs forward passes on dataset text under named component ablations,
computes KL, reverse KL, JSD, and CE diff per token, writes JSON.

Usage:
    python -m spd.editing.generate_token_divergence \\
        wandb:goodfire/spd/s-892f140b \\
        --edits edits.yaml \\
        --n_tokens 1500 \\
        --out_path /path/to/www/data/kl_tokens.json

edits.yaml format:
    Male pronouns:
      - h.1.mlp.down_proj:798
      - h.1.mlp.c_fc:144
      - h.1.attn.o_proj:82
    Question marks:
      - h.1.mlp.down_proj:534
"""

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from datasets import load_dataset

from spd.app.backend.app_tokenizer import AppTokenizer
from spd.editing import EditableModel, ForwardFn
from spd.settings import SPD_OUT_DIR

TokenData = dict[str, Any]


def compute_token_divergence(
    em: EditableModel,
    edit_fn: ForwardFn,
    token_ids: list[int],
    tok: AppTokenizer,
    top_k: int = 5,
) -> list[TokenData]:
    tokens = torch.tensor(token_ids, device="cuda")
    spans = tok.get_spans(token_ids)

    with torch.no_grad():
        bl_logits = em(tokens)
        ed_logits = edit_fn(tokens)

    bl_probs = F.softmax(bl_logits, dim=-1)
    ed_probs = F.softmax(ed_logits, dim=-1)
    bl_lp = F.log_softmax(bl_logits, dim=-1)
    ed_lp = F.log_softmax(ed_logits, dim=-1)

    # All metrics at positions [0..seq-2], predicting tokens [1..seq-1]
    fwd_kl_per_vocab = bl_probs[:-1] * (bl_lp[:-1] - ed_lp[:-1])
    fwd_kl = fwd_kl_per_vocab.sum(dim=-1)
    rev_kl = (ed_probs[:-1] * (ed_lp[:-1] - bl_lp[:-1])).sum(dim=-1)

    m_probs = 0.5 * (bl_probs[:-1] + ed_probs[:-1])
    m_lp = m_probs.log()
    jsd = 0.5 * (bl_probs[:-1] * (bl_lp[:-1] - m_lp)).sum(-1) + 0.5 * (
        ed_probs[:-1] * (ed_lp[:-1] - m_lp)
    ).sum(-1)

    targets = tokens[1:]
    ce_diff = -ed_lp[:-1][range(len(targets)), targets] - (
        -bl_lp[:-1][range(len(targets)), targets]
    )

    result: list[TokenData] = []
    for i in range(len(tokens)):
        if i == 0:
            result.append(
                {"s": spans[i], "kl": 0, "rkl": 0, "jsd": 0, "ce": 0, "bl": [], "ed": [], "kc": []}
            )
            continue

        prev = i - 1
        bl_top_v, bl_top_i = bl_probs[prev].topk(top_k)
        ed_top_v, ed_top_i = ed_probs[prev].topk(top_k)

        bl_top = [
            [tok.decode([int(t)]), round(v.item(), 4)]
            for v, t in zip(bl_top_v, bl_top_i, strict=True)
        ]
        ed_top = [
            [tok.decode([int(t)]), round(v.item(), 4)]
            for v, t in zip(ed_top_v, ed_top_i, strict=True)
        ]

        kl_contribs = fwd_kl_per_vocab[prev]
        _, kl_top_i = kl_contribs.abs().topk(top_k)
        kl_top = [
            [
                tok.decode([int(idx)]),
                round(bl_probs[prev, idx].item(), 4),
                round(ed_probs[prev, idx].item(), 4),
                round(kl_contribs[idx].item(), 5),
            ]
            for idx in kl_top_i
        ]

        result.append(
            {
                "s": spans[i],
                "kl": round(fwd_kl[prev].item(), 5),
                "rkl": round(rev_kl[prev].item(), 5),
                "jsd": round(jsd[prev].item(), 5),
                "ce": round(ce_diff[prev].item(), 5),
                "bl": bl_top,
                "ed": ed_top,
                "kc": kl_top,
            }
        )

    return result


def load_stories(n_tokens: int, max_seq_len: int = 300) -> list[list[int]]:
    """Load stories from SimpleStories until we have >= n_tokens."""
    ds = load_dataset("SimpleStories/SimpleStories", split="train", streaming=True)
    tok = AppTokenizer.from_pretrained("goodfire/SimpleStories-Llama-tokenizer")
    stories = []
    total = 0
    for item in ds:
        token_ids = tok.encode(item["story"])
        if len(token_ids) > max_seq_len:
            token_ids = token_ids[:max_seq_len]
        stories.append(token_ids)
        total += len(token_ids)
        if total >= n_tokens:
            break
    return stories


def main(
    wandb_path: str,
    edits: str,
    n_tokens: int = 1500,
    out_path: str | None = None,
) -> None:
    edits_path = Path(edits)
    assert edits_path.exists(), f"Edits file not found: {edits_path}"
    with open(edits_path) as f:
        edits_config: dict[str, list[str]] = yaml.safe_load(f)

    if out_path is None:
        out_path = str(SPD_OUT_DIR / "www" / "data" / "kl_tokens.json")
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    em, tok = EditableModel.from_wandb(wandb_path)
    stories = load_stories(n_tokens)
    total_tokens = sum(len(s) for s in stories)
    print(f"Loaded {len(stories)} stories, {total_tokens} tokens")

    all_data: dict[str, Any] = {}
    for edit_name, component_keys in edits_config.items():
        edit_dict = {k: 0.0 for k in component_keys}
        edit_fn = em.make_edit_fn(edit_dict)

        edit_stories = []
        for story_ids in stories:
            tokens = compute_token_divergence(em, edit_fn, story_ids, tok)
            edit_stories.append(tokens)

        all_data[edit_name] = {"components": component_keys, "stories": edit_stories}
        print(f"  {edit_name}: done")

    # Global p99 scales
    def p99(vals: list[float]) -> float:
        s = sorted(vals)
        return s[int(0.99 * len(s))]

    def collect(key: str) -> list[float]:
        return [t[key] for e in all_data.values() for s in e["stories"] for t in s if t[key] != 0]

    all_data["_meta"] = {
        "kl_max": round(p99(collect("kl")), 4),
        "rkl_max": round(p99(collect("rkl")), 4),
        "jsd_max": round(p99(collect("jsd")), 4),
        "ce_max": round(p99([abs(v) for v in collect("ce")]), 4),
    }

    with open(out, "w") as f:
        json.dump(all_data, f, separators=(",", ":"))

    size_kb = out.stat().st_size / 1024
    print(f"Wrote {size_kb:.0f} KB to {out}")


if __name__ == "__main__":
    import fire

    fire.Fire(main)
