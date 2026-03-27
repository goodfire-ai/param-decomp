# %% imports
import json
import random
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

from spd.editing.component_trainer import u_replaced
from spd.editing.generate_pareto_plots import (
    fire_positions,
    fire_set,
    get_examples,
    get_probs,
    kl_per_token,
    make_train_seqs,
    pad_train_seqs,
)
from spd.editing.lora_baseline import LoRATrainer
from spd.editing.utils import load_model
from spd.harvest.repo import HarvestRepo

WANDB_PATH = "wandb:goodfire/spd/s-55ea3f9b"
RUN_ID = "s-55ea3f9b"
MODULE_NAME = "h.2.mlp.down_proj"
U_IDX = 2359
TARGET_TOKEN = 80  # "o"

OUT_DIR = Path("/mnt/polished-lake/home/oli/spd/figures/editing")

# %% load
model, tok, config, dl = load_model(WANDB_PATH, device="cuda", batch_size=40)
harvest = HarvestRepo.open_most_recent(RUN_ID)
assert harvest is not None

examples = get_examples(harvest)
random.seed(42)
random.shuffle(examples)

eval_examples = examples[:50]
train_pool = examples[50:]

eval_tokens = [torch.tensor(ex.token_ids, device="cuda") for ex in eval_examples]
eval_baselines = get_probs(model, eval_tokens)

batch_sequences = next(iter(dl))["input_ids"]
global_tokens = [row.cuda() for row in batch_sequences]
global_baselines = get_probs(model, global_tokens)

lm_head = model.target_model.lm_head
assert isinstance(lm_head, torch.nn.Linear)
unembed = lm_head.weight[TARGET_TOKEN].detach().float()
assert unembed.ndim == 1
unembed_normed = unembed / unembed.norm()

print(f"Loaded: {len(examples)} examples, {len(eval_examples)} eval, {len(train_pool)} train")

# %% helpers
ForwardFn = Callable[[Tensor], Tensor]


def eval_edit(forward_fn: ForwardFn) -> tuple[list[float], list[float], list[float]]:
    kl_surr, p_fire, kl_global = [], [], []
    with torch.no_grad():
        for ex, tokens_t, base in zip(eval_examples, eval_tokens, eval_baselines, strict=True):
            fires = fire_set(ex)
            probs = forward_fn(tokens_t.unsqueeze(0))[0].softmax(-1)
            kl = kl_per_token(probs, base)
            for pos in fire_positions(ex):
                p_fire.append(probs[pos, TARGET_TOKEN].item())
            for i in range(len(tokens_t)):
                if i not in fires:
                    kl_surr.append(kl[i].item())
        for tokens_t, base in zip(global_tokens, global_baselines, strict=True):
            probs = forward_fn(tokens_t.unsqueeze(0))[0].softmax(-1)
            kl_global.extend(kl_per_token(probs, base).tolist())
    return kl_surr, p_fire, kl_global


# %% spd-sweep
alphas = [1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0]
spd_results = {}

for alpha in alphas:
    new_u = (-alpha * unembed_normed).to(torch.bfloat16)
    with u_replaced(model, MODULE_NAME, U_IDX, new_u) as fwd:
        spd_results[alpha] = eval_edit(fwd)
    kl_surr, p_fire, kl_global = spd_results[alpha]
    print(
        f"SPD α={alpha}: P('o')={np.mean(p_fire):.1%}, surr={np.mean(kl_surr):.6f}, global={np.mean(kl_global):.6f}"
    )

# %% lora-sweep
lora_lambdas = [0.1, 1.0, 10.0, 100.0]
BATCH_SIZE = 256
N_STEPS = 300


def get_lora_results(
    seqs: list[tuple[torch.Tensor, list[int]]],
) -> dict[float, tuple[list[float], list[float], list[float]]]:
    baselines = get_probs(model, [t for t, _ in seqs])
    all_tokens, all_baselines, all_fire, all_pad = pad_train_seqs(seqs, baselines)
    n = all_tokens.shape[0]

    lora_results = {}
    for lam in lora_lambdas:
        with LoRATrainer(model.target_model, MODULE_NAME, kl_weight=lam, lr=1e-3) as lora:
            for _ in range(N_STEPS):
                idxs = torch.randint(n, (min(BATCH_SIZE, n),))
                lora.train_step(
                    all_tokens[idxs],
                    all_baselines[idxs],
                    all_fire[idxs],
                    all_pad[idxs],
                )
            lora_results[lam] = eval_edit(lora.forward)
            kl_surr, p_fire, kl_global = lora_results[lam]
            print(
                f"LoRA λ={lam} (n={n}): P('o')={np.mean(p_fire):.1%}, surr={np.mean(kl_surr):.6f}, global={np.mean(kl_global):.6f}"
            )

    return lora_results


train_seqs = make_train_seqs(train_pool)
lora_results = get_lora_results(train_seqs)

train_seqs_low = make_train_seqs(train_pool[:10])
lora_results_low = get_lora_results(train_seqs_low)

# %% export
OUT_DIR.mkdir(parents=True, exist_ok=True)

sa = sorted(a for a in spd_results if a >= 1.5)
la = sorted(lam for lam in lora_results if lam <= 100.0)
la_ld = sorted(lam for lam in lora_results_low if lam <= 100.0)

pareto_data = {
    "spd": {
        str(a): {
            "p_fire": float(np.mean(spd_results[a][1])),
            "surr_kl": float(np.mean(spd_results[a][0])),
            "global_kl": float(np.mean(spd_results[a][2])),
        }
        for a in sa
    },
    "lora": {
        str(lam): {
            "p_fire": float(np.mean(lora_results[lam][1])),
            "surr_kl": float(np.mean(lora_results[lam][0])),
            "global_kl": float(np.mean(lora_results[lam][2])),
        }
        for lam in la
    },
    "lora_low": {
        str(lam): {
            "p_fire": float(np.mean(lora_results_low[lam][1])),
            "surr_kl": float(np.mean(lora_results_low[lam][0])),
            "global_kl": float(np.mean(lora_results_low[lam][2])),
        }
        for lam in la_ld
    },
    "meta": {
        "n_eval": len(eval_examples),
        "n_train": len(train_seqs),
        "n_train_low": 10,
        "n_global": len(global_tokens),
    },
}
(OUT_DIR / "pareto_data.json").write_text(json.dumps(pareto_data, indent=2))
print(f"Wrote {OUT_DIR / 'pareto_data.json'}")

# %% pareto-plot
fig, (ax_s, ax_g) = plt.subplots(1, 2, figsize=(14, 6))

for ax, idx, xlabel in [(ax_s, 0, "Surrounding KL"), (ax_g, 2, "Global KL")]:
    color = "#58a6ff"
    spd_kl = [np.mean(spd_results[a][idx]) for a in sa]
    spd_p = [np.mean(spd_results[a][1]) for a in sa]
    ax.plot(
        spd_kl,
        spd_p,
        "D-",
        color=color,
        linewidth=2,
        markersize=7,
        label="SPD analytical (sweep α)",
        zorder=5,
    )
    for a, x, y in zip(sa, spd_kl, spd_p, strict=True):
        ax.annotate(
            f"α={a}", (x, y), textcoords="offset points", xytext=(6, -4), fontsize=7, color=color
        )

    color_lora = "#f0883e"
    lora_kl = [np.mean(lora_results[lam][idx]) for lam in la]
    lora_p = [np.mean(lora_results[lam][1]) for lam in la]
    ax.plot(
        lora_kl,
        lora_p,
        "o-",
        color=color_lora,
        linewidth=2,
        markersize=7,
        label=f"LoRA n={len(train_seqs)} (sweep λ)",
        zorder=5,
    )
    for lam, x, y in zip(la, lora_kl, lora_p, strict=True):
        ax.annotate(
            f"λ={lam}",
            (x, y),
            textcoords="offset points",
            xytext=(6, -4),
            fontsize=7,
            color=color_lora,
        )

    color_ld = "#888888"
    lora_kl_ld = [np.mean(lora_results_low[lam][idx]) for lam in la_ld]
    lora_p_ld = [np.mean(lora_results_low[lam][1]) for lam in la_ld]
    ax.plot(
        lora_kl_ld,
        lora_p_ld,
        "s-",
        color=color_ld,
        linewidth=2,
        markersize=7,
        label="LoRA n=10 (sweep λ)",
        zorder=5,
    )
    for lam, x, y in zip(la_ld, lora_kl_ld, lora_p_ld, strict=True):
        ax.annotate(
            f"λ={lam}",
            (x, y),
            textcoords="offset points",
            xytext=(6, -4),
            fontsize=7,
            color=color_ld,
        )

    ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("P('o') at fire positions")
    ax.legend(fontsize=8)

fig.suptitle("Pareto: SPD vs LoRA baseline", fontsize=13)
fig.tight_layout()
fig.savefig(OUT_DIR / "pareto.png", dpi=150, bbox_inches="tight")
fig.savefig(OUT_DIR / "pareto.pdf", bbox_inches="tight")
plt.show()
print("Saved pareto.png and pareto.pdf")

# %% histogram-plot
best_alpha = min(sa, key=lambda a: abs(np.mean(spd_results[a][1]) - 0.95))
best_lambda = min(la, key=lambda lam: abs(np.mean(lora_results[lam][1]) - 0.95))
print(
    f"Matched at ~95%: SPD α={best_alpha} ({np.mean(spd_results[best_alpha][1]):.1%}), LoRA λ={best_lambda} ({np.mean(lora_results[best_lambda][1]):.1%})"
)

fig, (ax_s, ax_g) = plt.subplots(1, 2, figsize=(14, 6))
bins = np.logspace(-6, 2, 80)

spd_kl_s, spd_p, spd_kl_g = spd_results[best_alpha]
lora_kl_s, lora_p, lora_kl_g = lora_results[best_lambda]

ax_s.hist(
    spd_kl_s,
    bins=bins,
    color="#58a6ff",
    histtype="step",
    linewidth=1.5,
    label=f"SPD α={best_alpha} (P('o')={np.mean(spd_p):.0%})",
)
ax_s.hist(
    lora_kl_s,
    bins=bins,
    color="#f0883e",
    histtype="step",
    linewidth=1.5,
    label=f"LoRA λ={best_lambda} (P('o')={np.mean(lora_p):.0%})",
)
ax_s.set_title("Surrounding tokens")

ax_g.hist(
    spd_kl_g,
    bins=bins,
    color="#58a6ff",
    histtype="step",
    linewidth=1.5,
    label=f"SPD α={best_alpha}",
)
ax_g.hist(
    lora_kl_g,
    bins=bins,
    color="#f0883e",
    histtype="step",
    linewidth=1.5,
    label=f"LoRA λ={best_lambda}",
)
ax_g.set_title("Global tokens")

for ax in [ax_s, ax_g]:
    ax.set_xscale("log")
    ax.set_xlabel("KL divergence per token")
    ax.set_ylabel("Count")
    ax.legend(fontsize=9)

fig.suptitle("Per-token KL at matched P('o')", fontsize=13)
fig.tight_layout()
fig.savefig(OUT_DIR / "kl_histogram.png", dpi=150, bbox_inches="tight")
fig.savefig(OUT_DIR / "kl_histogram.pdf", bbox_inches="tight")
plt.show()
print("Saved kl_histogram.png and kl_histogram.pdf")
