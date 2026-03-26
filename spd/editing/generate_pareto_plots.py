"""Generate Pareto frontier plots: SPD component editing vs LoRA baseline.

Sweeps SPD analytical (α), SPD trained (n), and LoRA (n × λ), then plots
P('o') vs surrounding/global KL for all examples, emoticon-only, and non-emoticon.

Usage:
    python -m spd.editing.generate_pareto_plots --out-dir figures/
"""

import json
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import anthropic
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

from spd.editing.component_trainer import u_replaced
from spd.editing.lora_baseline import LoRATrainer
from spd.editing.utils import load_model
from spd.harvest.repo import HarvestRepo
from spd.harvest.schemas import ActivationExample
from spd.models.component_model import ComponentModel

WANDB_PATH = "wandb:goodfire/spd/s-55ea3f9b"
RUN_ID = "s-55ea3f9b"
MODULE_NAME = "h.2.mlp.down_proj"
U_IDX = 2359
TARGET_TOKEN = 80  # "o"

ForwardFn = Callable[[Tensor], Tensor]


def fire_positions(ex: ActivationExample) -> list[int]:
    """Positions where the component fires and a next-token exists."""
    return [i for i, f in enumerate(ex.firings) if f and i + 1 < len(ex.token_ids)]


def fire_set(ex: ActivationExample) -> set[int]:
    return {i for i, f in enumerate(ex.firings) if f}


@dataclass
class BlastRadius:
    on_fire_p: list[float] = field(default_factory=list)
    surrounding_kl: list[float] = field(default_factory=list)
    global_kl: list[float] = field(default_factory=list)


@dataclass
class ParetoPoint:
    p_all: float
    p_emo: float
    p_non_emo: float
    surr_kl: float
    glob_kl: float


def kl_per_token(probs_a: Tensor, probs_b: Tensor) -> Tensor:
    return (probs_a * ((probs_a + 1e-10).log() - (probs_b + 1e-10).log())).sum(-1)


def get_probs(component_model: ComponentModel, seqs: list[Tensor]) -> list[Tensor]:
    with torch.no_grad():
        return [component_model(t.unsqueeze(0))[0].softmax(-1) for t in seqs]


def get_probs_raw(forward_fn: ForwardFn, seqs: list[Tensor]) -> list[Tensor]:
    with torch.no_grad():
        return [forward_fn(t.unsqueeze(0))[0].softmax(-1) for t in seqs]


def measure_blast(
    forward_fn: ForwardFn,
    examples: list[ActivationExample],
    tokens: list[Tensor],
    baselines: list[Tensor],
    global_tokens: list[Tensor],
    global_baselines: list[Tensor],
) -> BlastRadius:
    r = BlastRadius()
    with torch.no_grad():
        for ex, tokens_t, base in zip(examples, tokens, baselines, strict=True):
            probs = forward_fn(tokens_t.unsqueeze(0))[0].softmax(-1)
            kl = kl_per_token(probs, base)
            fires = fire_set(ex)
            for pos in fire_positions(ex):
                r.on_fire_p.append(probs[pos, TARGET_TOKEN].item())
            for i in range(kl.shape[0]):
                if i not in fires:
                    r.surrounding_kl.append(kl[i].item())
        for tokens_t, base in zip(global_tokens, global_baselines, strict=True):
            probs = forward_fn(tokens_t.unsqueeze(0))[0].softmax(-1)
            kl = kl_per_token(probs, base)
            r.global_kl.extend(kl.tolist())
    return r


def measure_pareto(
    forward_fn: ForwardFn,
    all_examples: list[ActivationExample],
    all_tokens: list[Tensor],
    eval_baselines: list[Tensor],
    emo_idxs: list[int],
    non_emo_idxs: list[int],
    global_tokens: list[Tensor],
    global_baselines: list[Tensor],
) -> ParetoPoint:
    br = measure_blast(
        forward_fn,
        all_examples,
        all_tokens,
        eval_baselines,
        global_tokens,
        global_baselines,
    )
    br_emo = measure_blast(
        forward_fn,
        [all_examples[i] for i in emo_idxs],
        [all_tokens[i] for i in emo_idxs],
        [eval_baselines[i] for i in emo_idxs],
        [],
        [],
    )
    br_non = measure_blast(
        forward_fn,
        [all_examples[i] for i in non_emo_idxs],
        [all_tokens[i] for i in non_emo_idxs],
        [eval_baselines[i] for i in non_emo_idxs],
        [],
        [],
    )
    return ParetoPoint(
        p_all=float(np.mean(br.on_fire_p)),
        p_emo=float(np.mean(br_emo.on_fire_p)),
        p_non_emo=float(np.mean(br_non.on_fire_p)),
        surr_kl=float(np.mean(br.surrounding_kl)),
        glob_kl=float(np.mean(br.global_kl)),
    )


def get_examples(harvest: HarvestRepo) -> list[ActivationExample]:
    """Examples with at least one firing position (and a next token after it)."""
    comp = harvest.get_component(f"{MODULE_NAME}:{U_IDX}")
    assert comp is not None
    return [ex for ex in comp.activation_examples if fire_positions(ex)]


def make_train_seqs(
    examples: list[ActivationExample],
) -> list[tuple[Tensor, list[int]]]:
    """Build (mutated_tokens, fire_positions) pairs for training."""
    seqs = []
    for ex in examples:
        t = torch.tensor(ex.token_ids, device="cuda")
        positions = fire_positions(ex)
        for pos in positions:
            t[pos + 1] = TARGET_TOKEN
        seqs.append((t, positions))
    return seqs


def label_emoticon_examples(
    examples: list[ActivationExample],
    tok_display_fn: Callable[[list[int]], list[str]],
) -> tuple[list[int], list[int]]:
    """Use Claude haiku to classify examples as emoticon vs not. Returns (emo_idxs, non_emo_idxs)."""
    contexts = []
    for i, ex in enumerate(examples):
        spans = tok_display_fn(ex.token_ids)
        positions = fire_positions(ex)
        snippets = []
        for p in positions:
            ctx_start, ctx_end = max(0, p - 4), min(len(spans), p + 4)
            snippet = "".join(spans[ctx_start:ctx_end]).strip()
            fire_tok = spans[p].strip()
            next_tok = spans[p + 1].strip() if p + 1 < len(spans) else ""
            snippets.append(f'"{snippet}" ["{fire_tok}"→"{next_tok}"]')
        contexts.append(f"{i}: {'; '.join(snippets)}")

    client = anthropic.Anthropic()  # pyright: ignore[reportAttributeAccessIssue]
    resp = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=4096,
        temperature=0.0,
        messages=[
            {
                "role": "user",
                "content": f"""Classify each example as EMOTICON (true) or NOT (false). An example is EMOTICON if ANY of its firing positions start an emoticon like :) ;-) :D :-( :P :o) >:( =) :3 :] :/ etc. NOT: code, URLs, timestamps, package names, TL;DR, :class:, line numbers, regular punctuation.
Return ONLY a JSON object mapping id to true/false.

{chr(10).join(contexts)}""",
            }
        ],
    )
    text: str = resp.content[0].text
    labels = json.loads(text[text.index("{") : text.rindex("}") + 1])
    emo = [int(k) for k, v in labels.items() if v]
    non_emo = [int(k) for k, v in labels.items() if not v]
    return emo, non_emo


def plot_pareto(pareto_data: dict[str, ParetoPoint], lora_ns: list[int], out_dir: Path) -> None:
    kl_weights = [0.0, 1.0, 3.0, 10.0, 30.0, 100.0]
    alphas = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0]
    max_lora_n = max(lora_ns)

    # Main plot: SPD analytical (sweep α) vs LoRA max-n (sweep λ)
    fig, (ax_s, ax_g) = plt.subplots(1, 2, figsize=(14, 6))

    for ax, kl_field, xlabel in [
        (ax_s, "surr_kl", "Surrounding KL"),
        (ax_g, "glob_kl", "Global KL"),
    ]:
        # SPD analytical
        ana_kl = [getattr(pareto_data[f"spd_analytical_a{a}"], kl_field) for a in alphas]
        ana_p = [pareto_data[f"spd_analytical_a{a}"].p_all for a in alphas]
        ax.plot(
            ana_kl,
            ana_p,
            "D-",
            color="#58a6ff",
            linewidth=2,
            markersize=7,
            label="SPD analytical (sweep α)",
            zorder=5,
        )
        for a, x, y in zip(alphas, ana_kl, ana_p, strict=True):
            ax.annotate(
                f"α={a}",
                (x, y),
                textcoords="offset points",
                xytext=(6, -4),
                fontsize=7,
                color="#58a6ff",
            )

        # LoRA max-n
        lora_kl = [getattr(pareto_data[f"lora_n{max_lora_n}_l{w}"], kl_field) for w in kl_weights]
        lora_p = [pareto_data[f"lora_n{max_lora_n}_l{w}"].p_all for w in kl_weights]
        ax.plot(
            lora_kl,
            lora_p,
            "o-",
            color="#f0883e",
            linewidth=2,
            markersize=7,
            label=f"LoRA n={max_lora_n} (sweep λ)",
            zorder=5,
        )
        for w, x, y in zip(kl_weights, lora_kl, lora_p, strict=True):
            ax.annotate(
                f"λ={w}",
                (x, y),
                textcoords="offset points",
                xytext=(6, -4),
                fontsize=7,
                color="#f0883e",
            )

        ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("P('o') at fire positions")
        ax.legend(fontsize=9)

    fig.suptitle("Pareto: SPD analytical vs LoRA baseline", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "pareto_main.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved pareto_main.png")


def main(out_dir: Path) -> None:
    out_dir.mkdir(exist_ok=True)
    model, tok, _, dl = load_model(WANDB_PATH, device="cuda", batch_size=40)
    harvest = HarvestRepo.open_most_recent(RUN_ID)
    assert harvest is not None

    examples = get_examples(harvest)
    random.seed(42)
    random.shuffle(examples)

    # Eval: first 50 examples, train: rest
    eval_examples = examples[:50]
    train_pool = examples[50:]
    eval_tokens = [torch.tensor(ex.token_ids, device="cuda") for ex in eval_examples]

    global_tokens = [row.cuda() for row in next(iter(dl))["input_ids"]]

    # LLM-label eval examples
    print("Labeling eval examples...")
    emo_idxs, non_emo_idxs = label_emoticon_examples(eval_examples, tok.get_spans)
    print(f"  {len(emo_idxs)} emoticon, {len(non_emo_idxs)} non-emoticon")

    pareto_data: dict[str, ParetoPoint] = {}

    # --- Baselines ---
    lm_head = model.target_model.lm_head
    assert isinstance(lm_head, torch.nn.Linear)
    unembed = lm_head.weight[TARGET_TOKEN].detach().float()
    unembed_normed = unembed / unembed.norm()

    eval_baselines = get_probs(model, eval_tokens)
    global_baselines = get_probs(model, global_tokens)

    pareto_eval_args = (
        eval_examples,
        eval_tokens,
        eval_baselines,
        emo_idxs,
        non_emo_idxs,
        global_tokens,
        global_baselines,
    )

    # --- SPD analytical ---
    for alpha in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0]:
        new_u = (-alpha * unembed_normed).to(torch.bfloat16)
        with u_replaced(model, MODULE_NAME, U_IDX, new_u) as fwd:
            pareto_data[f"spd_analytical_a{alpha}"] = measure_pareto(fwd, *pareto_eval_args)
    print("SPD analytical done")

    # --- LoRA ---
    kl_weights = [0.0, 1.0, 3.0, 10.0, 30.0, 100.0]
    lora_ns = [1, 8, 64, 256, len(train_pool)]

    def forward_base(tokens: Tensor) -> Tensor:
        return model.target_model(tokens)[0]

    for n_ex in lora_ns:
        train_seqs = make_train_seqs(train_pool[:n_ex])
        # Cache baselines before any LoRA hook
        train_baselines = get_probs_raw(forward_base, [t for t, _ in train_seqs])

        for kl_w in kl_weights:
            with LoRATrainer(model.target_model, MODULE_NAME, lr=1e-3, kl_weight=kl_w) as lora:
                for _ in range(300):
                    idx = random.randint(0, len(train_seqs) - 1)
                    lora.train_step(train_seqs[idx], train_baselines[idx])
                pareto_data[f"lora_n{n_ex}_l{kl_w}"] = measure_pareto(
                    lora.forward, *pareto_eval_args
                )

        r = pareto_data[f"lora_n{n_ex}_l10.0"]
        print(f"  LoRA n={n_ex} λ=10: P_emo={r.p_emo:.0%} surr={r.surr_kl:.4f}")

    print(f"\n{len(pareto_data)} pareto points. Plotting...")
    plot_pareto(pareto_data, lora_ns, out_dir)

    # Export raw data
    from dataclasses import asdict

    data_out = {k: asdict(v) for k, v in pareto_data.items()}
    data_out["_meta"] = {
        "eval_n": len(eval_examples),
        "train_n": len(train_pool),
        "emo_n": len(emo_idxs),
        "non_emo_n": len(non_emo_idxs),
        "global_n": len(global_tokens),
        "lora_ns": lora_ns,
    }
    (out_dir / "pareto_data.json").write_text(json.dumps(data_out, indent=2))
    print(f"Data written to {out_dir / 'pareto_data.json'}")
    print("Done.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    main(args.out_dir)
