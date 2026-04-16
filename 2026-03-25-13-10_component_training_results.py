# %% md-intro
# # Write-Vector Training: Redirecting SPD Component Output
#
# **Claim**: Given an SPD decomposition, we can train a single component's write vector
# (U row) to predict a different token at firing positions, with minimal collateral damage
# to the rest of the model's behavior.
#
# **Setup**: Jose (`s-55ea3f9b`), `pile_llama_simple_mlp-4L` trained on The Pile.
# We find 7 emoticon-related components in `h.2.mlp.down_proj` via autointerp label search,
# then train each one's write vector to predict `'o'` (token 80) at firing positions.

# %% setup
import random
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython.display import HTML, display
from jaxtyping import Float, Int
from spd.editing.compare import ExampleDiff, TokenDiff
from spd.editing.component_trainer import ComponentTrainer
from torch import Tensor

from spd.autointerp.repo import InterpRepo
from spd.editing import (
    EditableModel,
    parse_component_key,
    render_edit_comparison,
    search_interpretations,
)
from spd.harvest.repo import HarvestRepo
from spd.harvest.schemas import ActivationExample

JOSE = "wandb:goodfire/spd/s-55ea3f9b"
JOSE_ID = "s-55ea3f9b"
TARGET_TOKEN = 80  # "o"

em, tok = EditableModel.from_wandb(JOSE)
harvest = HarvestRepo.open_most_recent(JOSE_ID)
interp = InterpRepo.open(JOSE_ID)
assert harvest and interp
print(f"Model loaded. {len(em.model.components)} decomposed layers.")


# %% helpers
def get_firing_positions(comp_key: str) -> list[tuple[ActivationExample, int]]:
    comp = harvest.get_component(comp_key)
    assert comp is not None
    result = []
    for ex in comp.activation_examples:
        for i, fires in enumerate(ex.firings):
            if fires and i + 1 < len(ex.token_ids):
                result.append((ex, i))
    return result


def make_train_seqs(
    examples: list[tuple[ActivationExample, int]],
    target_token: int = TARGET_TOKEN,
) -> list[tuple[Int[Tensor, " seq"], list[int]]]:
    seqs = []
    for ex, p in examples:
        t = torch.tensor(ex.token_ids, device="cuda")
        t[p + 1] = target_token
        seqs.append((t, [p]))
    return seqs


def make_eval_seqs(
    examples: list[tuple[ActivationExample, int]],
) -> tuple[list[Int[Tensor, " seq"]], list[list[int]], list[set[int]]]:
    """Returns (token_seqs, positions_per_seq, fire_sets_per_seq).

    fire_sets include ALL firing positions in each sequence (not just the eval target),
    so surrounding/post-fire metrics can exclude other firings.
    """
    tokens, positions, fire_sets = [], [], []
    for ex, p in examples:
        tokens.append(torch.tensor(ex.token_ids, device="cuda"))
        positions.append([p])
        fire_sets.append({i for i, f in enumerate(ex.firings) if f})
    return tokens, positions, fire_sets


def get_global_seqs(n: int = 40) -> list[Int[Tensor, " seq"]]:
    comp = harvest.get_component("h.0.mlp.c_fc:100")
    assert comp is not None
    return [torch.tensor(ex.token_ids, device="cuda") for ex in comp.activation_examples[:n]]


def kl_per_token(
    probs_edit: Float[Tensor, "seq vocab"],
    probs_base: Float[Tensor, "seq vocab"],
) -> Float[Tensor, " seq"]:
    return (probs_edit * ((probs_edit + 1e-10).log() - (probs_base + 1e-10).log())).sum(-1)


def train_write_vector(
    comp_key: str,
    train_seqs: list[tuple[Int[Tensor, " seq"], list[int]]],
    n_steps: int = 100,
    lr: float = 1e-3,
) -> tuple[ComponentTrainer, EditableModel]:
    """Train a component's write vector. Returns (trainer, em) with trainer still alive."""
    em_t, _ = EditableModel.from_wandb(JOSE)
    trainer = ComponentTrainer(em_t.model, targets={comp_key: "write"}, lr=lr)
    for _ in range(n_steps):
        for tokens_mut, positions in train_seqs:
            logits = trainer(tokens_mut.unsqueeze(0))
            pos_t = torch.tensor(positions, device=tokens_mut.device)
            loss = torch.nn.functional.cross_entropy(logits[0, positions], tokens_mut[pos_t + 1])
            trainer.step(loss)
    return trainer, em_t


def cache_baselines(
    trainer: ComponentTrainer,
    *seq_lists: list[Int[Tensor, " seq"]],
) -> dict[int, Float[Tensor, "seq vocab"]]:
    baselines = {}
    with torch.no_grad():
        for seq_list in seq_lists:
            for tokens_t in seq_list:
                baselines[id(tokens_t)] = trainer(tokens_t.unsqueeze(0))[0].softmax(-1)
    return baselines


@dataclass
class BlastRadiusResult:
    on_fire_ranks: list[int] = field(default_factory=list)
    on_fire_p: list[float] = field(default_factory=list)
    post_fire_kl: list[float] = field(default_factory=list)
    surrounding_kl: list[float] = field(default_factory=list)
    global_kl: list[float] = field(default_factory=list)


def measure_blast_radius(
    trainer: ComponentTrainer,
    baselines: dict[int, Float[Tensor, "seq vocab"]],
    eval_tokens: list[Int[Tensor, " seq"]],
    eval_positions: list[list[int]],
    eval_fire_sets: list[set[int]],
    global_tokens: list[Int[Tensor, " seq"]],
    target_token: int = TARGET_TOKEN,
) -> BlastRadiusResult:
    r = BlastRadiusResult()
    with torch.no_grad():
        for tokens_t, positions, fire_set in zip(eval_tokens, eval_positions, eval_fire_sets):
            logits = trainer(tokens_t.unsqueeze(0))[0]
            probs = logits.softmax(-1)
            kl = kl_per_token(probs, baselines[id(tokens_t)])

            for p in positions:
                r.on_fire_ranks.append((logits[p] >= logits[p, target_token]).sum().item())
                r.on_fire_p.append(probs[p, target_token].item())
                for offset in range(1, 6):
                    pos = p + offset
                    if pos < kl.shape[0] and pos not in fire_set:
                        r.post_fire_kl.append(kl[pos].item())

            for i in range(kl.shape[0]):
                if i not in fire_set:
                    r.surrounding_kl.append(kl[i].item())

        for tokens_t in global_tokens:
            probs = trainer(tokens_t.unsqueeze(0))[0].softmax(-1)
            kl = kl_per_token(probs, baselines[id(tokens_t)])
            r.global_kl.extend(kl.tolist())

    return r


print("Helpers defined.")


# %% md-cluster
# ## 1. Finding emoticon components

# %% cluster
matches = search_interpretations(harvest, interp, r"emoticon|emoji|eyes|smiley|face")

current_module = None
for m in sorted(matches, key=lambda m: (-int(m.key.split('.')[1]), 'c_fc' not in m.key, parse_component_key(m.key)[1])):
    module, idx = parse_component_key(m.key)
    if module != current_module:
        current_module = module
        print(f"\n  {module}")
    print(f"    :{idx:4d}  {m.label}")

down_proj_keys = sorted([m.key for m in matches if "down_proj" in m.key])
print(f"\n{len(matches)} total, {len(down_proj_keys)} in h.2.mlp.down_proj (sweep targets)")


# %% md-sweep
# ## 2. Sweep: which component makes the best edit target?

# %% sweep
random.seed(42)
sweep_results = []
global_tokens = get_global_seqs()

for key in down_proj_keys:
    firings = get_firing_positions(key)
    if len(firings) < 10:
        print(f"  {key}: only {len(firings)} firings, skipping")
        continue

    random.shuffle(firings)
    train_seqs = make_train_seqs(firings[:6])
    eval_tokens, eval_positions, eval_fire_sets = make_eval_seqs(firings[6:36])

    trainer, em_k = train_write_vector(key, train_seqs)
    baselines = cache_baselines(trainer, eval_tokens, global_tokens)

    # Need to re-train since cache_baselines used the trainer before training
    # Actually — train_write_vector already trained. But cache_baselines needs
    # the BASELINE (pre-edit) probs. So we need a fresh trainer for baselines.
    # Let's fix: load fresh, cache, then train.
    trainer.cleanup()
    del em_k

    # Proper order: fresh model → cache baselines → train → eval
    em_k, _ = EditableModel.from_wandb(JOSE)
    trainer = ComponentTrainer(em_k.model, targets={key: "write"}, lr=1e-3)
    baselines = cache_baselines(trainer, eval_tokens, global_tokens)

    for _ in range(100):
        for tokens_mut, positions in train_seqs:
            logits = trainer(tokens_mut.unsqueeze(0))
            pos_t = torch.tensor(positions, device=tokens_mut.device)
            loss = torch.nn.functional.cross_entropy(logits[0, positions], tokens_mut[pos_t + 1])
            trainer.step(loss)

    br = measure_blast_radius(trainer, baselines, eval_tokens, eval_positions, eval_fire_sets, global_tokens)
    trainer.cleanup()
    del em_k

    label = next(m.label for m in matches if m.key == key)
    mean_p = np.mean(br.on_fire_p)
    mean_pfkl = np.mean(br.post_fire_kl) if br.post_fire_kl else 0.0
    mean_gkl = np.mean(br.global_kl)
    sweep_results.append((key, label, mean_p, mean_pfkl, mean_gkl))
    _, idx = parse_component_key(key)
    print(f"  :{idx:<5d}  P(o)={mean_p:.1%}  post-fire={mean_pfkl:.4f}  global={mean_gkl:.4f}")

# %% sweep-table
print(f"\n{'Component':>10s}  {'P(o)':>6s}  {'Post-fire KL':>12s}  {'Global KL':>10s}  Label")
print("-" * 100)
for key, label, prob, pfkl, gkl in sorted(sweep_results, key=lambda x: -x[2]):
    _, idx = parse_component_key(key)
    best_p = " ◀" if prob == max(r[2] for r in sweep_results) else ""
    best_l = " ◀" if pfkl == min(r[3] for r in sweep_results) else ""
    print(f"    :{idx:<5d}  {prob:>5.1%}  {pfkl:>12.4f}{best_l:3s}  {gkl:>10.4f}  {label}{best_p}")


# %% md-deep-dive
# ## 3. Deep dive: `:2359` collateral analysis

# %% deep-dive
key_2359 = "h.2.mlp.down_proj:2359"
module_2359, cidx_2359 = parse_component_key(key_2359)

firings_2359 = get_firing_positions(key_2359)
random.seed(42)
random.shuffle(firings_2359)

train_seqs = make_train_seqs(firings_2359[:6])
eval_tokens, eval_positions, eval_fire_sets = make_eval_seqs(firings_2359[6:36])

# Fresh model → baselines → train → eval
em_dd, tok_dd = EditableModel.from_wandb(JOSE)
trainer = ComponentTrainer(em_dd.model, targets={key_2359: "write"}, lr=1e-3)
baselines = cache_baselines(trainer, eval_tokens, global_tokens)

for _ in range(100):
    for tokens_mut, positions in train_seqs:
        logits = trainer(tokens_mut.unsqueeze(0))
        pos_t = torch.tensor(positions, device=tokens_mut.device)
        loss = torch.nn.functional.cross_entropy(logits[0, positions], tokens_mut[pos_t + 1])
        trainer.step(loss)

br = measure_blast_radius(trainer, baselines, eval_tokens, eval_positions, eval_fire_sets, global_tokens)

print("=== :2359 collateral analysis ===\n")
print(f"Held-out P('o'): {np.mean(br.on_fire_p):.1%}  (n={len(br.on_fire_p)})\n")
print(f"{'Position':>25s}  {'Mean KL':>8s}  {'n':>6s}")
print("-" * 50)
print(f"{'At firing':>25s}  {'—':>8s}  {len(br.on_fire_p):>6d}")
print(f"{'Post-fire (1-5)':>25s}  {np.mean(br.post_fire_kl):>8.4f}  {len(br.post_fire_kl):>6d}")
print(f"{'Surrounding (non-fire)':>25s}  {np.mean(br.surrounding_kl):>8.4f}  {len(br.surrounding_kl):>6d}")
print(f"{'Global (random text)':>25s}  {np.mean(br.global_kl):>8.4f}  {len(br.global_kl):>6d}")


# %% md-write-vector
# ## 4. Write vector analysis

# %% write-vector
em_orig, _ = EditableModel.from_wandb(JOSE)

u_trained = em_dd.model.components[module_2359].U[cidx_2359].detach().float()
u_original = em_orig.model.components[module_2359].U[cidx_2359].detach().float().clone()
v_trained = em_dd.model.components[module_2359].V[:, cidx_2359].detach().float()
v_original = em_orig.model.components[module_2359].V[:, cidx_2359].detach().float()

cos_u = torch.nn.functional.cosine_similarity(u_trained, u_original, dim=0).item()
cos_v = torch.nn.functional.cosine_similarity(v_trained, v_original, dim=0).item()

print("=== Write vector (U row) ===")
print(f"  Cosine sim with original: {cos_u:.3f}")
print(f"  Norm: {u_original.norm().item():.2f} → {u_trained.norm().item():.2f} ({u_trained.norm().item()/u_original.norm().item():.1f}x)")
print("\n=== Read vector (V column) ===")
print(f"  Cosine sim with original: {cos_v:.3f}")
print(f"  Norm delta: {(v_trained - v_original).norm().item():.6f}")

boosts, supps = em_dd.unembed_alignment(key_2359, tok_dd)
print("\nTop 5 boosted tokens (cosine with unembed):")
for m in boosts[:5]:
    print(f"  {m.token_str:>12s}  cos={m.cosine:+.3f}  dot={m.dot:+.3f}")
print("\nTop 5 suppressed tokens:")
for m in supps[:5]:
    print(f"  {m.token_str:>12s}  cos={m.cosine:+.3f}  dot={m.dot:+.3f}")


# %% md-viz
# ## 5. Visualization: per-token KL heatmap

# %% viz-heldout
def build_diffs(
    trainer: ComponentTrainer,
    baselines: dict[int, Float[Tensor, "seq vocab"]],
    tokens_list: list[Int[Tensor, " seq"]],
    positions_list: list[list[int]],
    em_ref: EditableModel,
    comp_key: str,
) -> list[ExampleDiff]:
    module, cidx = parse_component_key(comp_key)
    diffs = []
    with torch.no_grad():
        for tokens_t, positions in zip(tokens_list, positions_list):
            probs_edit = trainer(tokens_t.unsqueeze(0))[0].softmax(-1)
            probs_base = baselines[id(tokens_t)]
            kl = kl_per_token(probs_edit, probs_base)
            ci_vals = em_ref.get_ci(tokens_t)[module][:, cidx].cpu()
            act_vals = em_ref.get_component_activations(tokens_t, comp_key).cpu()
            spans = tok.get_spans(tokens_t.tolist())
            fire_set = set(positions)

            token_diffs = []
            for t in range(len(spans)):
                diff = probs_edit[t] - probs_base[t]
                inc_idx = diff.topk(5).indices
                dec_idx = (-diff).topk(5).indices
                before_idx = probs_base[t].topk(8).indices
                after_idx = probs_edit[t].topk(8).indices
                token_diffs.append(TokenDiff(
                    span=spans[t], kl=kl[t].item(), ci=ci_vals[t].item(),
                    activation=act_vals[t].item(), fires=t in fire_set,
                    topk_before=[(tok.get_tok_display(int(j)), probs_base[t, j].item()) for j in before_idx],
                    topk_after=[(tok.get_tok_display(int(j)), probs_edit[t, j].item()) for j in after_idx],
                    top_increases=[(tok.get_tok_display(int(j)), probs_base[t, j].item(), probs_edit[t, j].item()) for j in inc_idx],
                    top_decreases=[(tok.get_tok_display(int(j)), probs_base[t, j].item(), probs_edit[t, j].item()) for j in dec_idx],
                ))
            diffs.append(ExampleDiff(tokens=token_diffs, max_kl=kl.max().item()))
    diffs.sort(key=lambda d: -d.max_kl)
    return diffs


heldout_diffs = build_diffs(trainer, baselines, eval_tokens, eval_positions, em_dd, key_2359)
display(HTML(render_edit_comparison(
    heldout_diffs[:15],
    title="Held-out: sequences where :2359 fires",
    subtitle="KL concentrated at firing position (orange). Post-fire bleed ~0.01.",
)))

# %% viz-global
global_diffs = build_diffs(
    trainer, baselines, global_tokens[:15],
    [[] for _ in global_tokens[:15]], em_dd, key_2359,
)
display(HTML(render_edit_comparison(
    global_diffs,
    title="Global: random text (no emoticon firing)",
    subtitle="Mean KL ~0.007 per token. Edit has negligible effect on non-firing text.",
)))


# %% md-single-example
# ## 6. Single-example training: do different examples converge?

# %% single-example
firings_2359 = get_firing_positions(key_2359)
random.seed(42)
random.shuffle(firings_2359)

N_RUNS = 10
deltas = []
shared_heldout = [(torch.tensor(ex.token_ids, device="cuda"), [p]) for ex, p in firings_2359[N_RUNS:N_RUNS + 30]]

for run_i in range(N_RUNS):
    ex, p = firings_2359[run_i]
    t = torch.tensor(ex.token_ids, device="cuda")
    t[p + 1] = TARGET_TOKEN

    em_i, _ = EditableModel.from_wandb(JOSE)
    tr_i = ComponentTrainer(em_i.model, targets={key_2359: "write"}, lr=1e-3)

    for _ in range(100):
        logits = tr_i(t.unsqueeze(0))
        loss = torch.nn.functional.cross_entropy(logits[0, [p]], t[[p + 1]])
        tr_i.step(loss)

    with torch.no_grad():
        probs = [tr_i(s.unsqueeze(0))[0, pos].softmax(-1)[TARGET_TOKEN].item()
                 for s, positions in shared_heldout for pos in positions]

    delta = em_i.model.components[module_2359].U[cidx_2359].detach().float() - u_original
    deltas.append((delta.clone(), np.mean(probs)))
    tr_i.cleanup()
    del em_i
    print(f"  Run {run_i}: P('o')={deltas[-1][1]:.1%}  |delta|={delta.norm().item():.3f}")

# %% single-example-analysis
delta_vecs = torch.stack([d for d, _ in deltas])
delta_norms = delta_vecs.norm(dim=1, keepdim=True)
cos_matrix = (delta_vecs @ delta_vecs.T) / (delta_norms @ delta_norms.T + 1e-10)
n = len(deltas)
upper_tri = [cos_matrix[i, j].item() for i in range(n) for j in range(i + 1, n)]

print(f"=== Single-example delta convergence (N={N_RUNS}) ===\n")
print("Pairwise cosine similarity of U deltas:")
print(f"  Mean: {np.mean(upper_tri):.3f}  Min: {min(upper_tri):.3f}  Max: {max(upper_tri):.3f}  Std: {np.std(upper_tri):.3f}")

probs_all = [p for _, p in deltas]
print(f"\nHeld-out P('o'): Mean={np.mean(probs_all):.1%}  Min={min(probs_all):.1%}  Max={max(probs_all):.1%}")

mean_delta = delta_vecs.mean(dim=0)
mean_delta_normed = mean_delta / mean_delta.norm()
cos_with_mean = [(d / d.norm()) @ mean_delta_normed for d in delta_vecs]
print(f"\nCosine with mean delta: {' '.join(f'{c.item():.3f}' for c in cos_with_mean)}")


# %% md-unembed-alignment
# ## 7. Are deltas aligned with output embeddings?

# %% unembed-alignment
unembed = em_orig.model.target_model.lm_head.weight.detach().float()
target_unembed = unembed[TARGET_TOKEN]
target_unembed_normed = target_unembed / target_unembed.norm()

print("Cosine(delta, unembed('o')) per single-example run:")
for i, (delta, prob) in enumerate(deltas):
    cos = ((delta / delta.norm()) @ target_unembed_normed).item()
    print(f"  Run {i}: cos={cos:+.3f}  P('o')={prob:.1%}")

all_cos = (unembed / unembed.norm(dim=1, keepdim=True)) @ mean_delta_normed
rank_asc = (all_cos <= all_cos[TARGET_TOKEN]).sum().item()

print(f"\nMean delta vs all {unembed.shape[0]} unembed vectors:")
print(f"  cos(mean_delta, unembed('o')) = {all_cos[TARGET_TOKEN].item():+.4f}")
print(f"  Rank of 'o': {rank_asc}/{unembed.shape[0]} most anti-aligned")

print("\nTop 5 anti-aligned:")
for j in (-all_cos).topk(5).indices:
    print(f"  {tok.get_tok_display(int(j)):>12s}  cos={all_cos[j].item():+.4f}")

random_dirs = torch.randn(100, mean_delta.shape[0], device=mean_delta.device)
random_dirs = random_dirs / random_dirs.norm(dim=1, keepdim=True)
print(f"\nMax |cos| with any unembed: {all_cos.abs().max().item():.4f}")
print(f"Max |cos| with 100 random dirs: {(random_dirs @ mean_delta_normed).abs().max().item():.4f}")


# %% md-blast-radius
# ## 8. Edit blast radius: on-fire vs post-fire vs surrounding vs global
#
# Sweep over U-scale (edit strength) × n-examples (training data).
# Three off-target regimes:
# - **Post-fire**: positions 1-5 after a firing (excluding other firings)
# - **Surrounding**: all non-fire positions in activation windows
# - **Global**: random text

# %% blast-radius
firings_2359 = get_firing_positions(key_2359)
random.seed(42)
random.shuffle(firings_2359)

MAX_TRAIN = 32
eval_tokens, eval_positions, eval_fire_sets = make_eval_seqs(firings_2359[MAX_TRAIN:MAX_TRAIN + 50])

n_examples_list = [1, 2, 4, 8, 16, 32]
scales_br = np.linspace(0, 1.5, 31)
results_grid: dict[tuple[int, float], BlastRadiusResult] = {}

for n_ex in n_examples_list:
    train_seqs = make_train_seqs(firings_2359[:n_ex])

    # Fresh model → cache baselines → train → get delta
    em_br, _ = EditableModel.from_wandb(JOSE)
    tr_br = ComponentTrainer(em_br.model, targets={key_2359: "write"}, lr=1e-3)
    baselines_br = cache_baselines(tr_br, eval_tokens, global_tokens)

    for _ in range(100):
        for tokens_mut, positions in train_seqs:
            logits = tr_br(tokens_mut.unsqueeze(0))
            pos_t = torch.tensor(positions, device=tokens_mut.device)
            loss = torch.nn.functional.cross_entropy(logits[0, positions], tokens_mut[pos_t + 1])
            tr_br.step(loss)
    tr_br.cleanup()

    u_trained_br = em_br.model.components[module_2359].U[cidx_2359].detach().clone()
    delta_br = u_trained_br - u_original

    # Sweep scales: reset U, create one trainer, modify U per scale
    em_br.model.components[module_2359].U.data[cidx_2359] = u_original.clone()
    tr_sweep = ComponentTrainer(em_br.model, targets={key_2359: "write"}, lr=1e-3)

    for scale in scales_br:
        em_br.model.components[module_2359].U.data[cidx_2359] = u_original + scale * delta_br
        br = measure_blast_radius(
            tr_sweep, baselines_br, eval_tokens, eval_positions, eval_fire_sets, global_tokens,
        )
        results_grid[(n_ex, scale)] = br

    tr_sweep.cleanup()
    del em_br

    r = results_grid[(n_ex, 1.0)]
    print(f"  n={n_ex:>2d}  P('o')={np.mean(r.on_fire_p):.1%}  med_rank={np.median(r.on_fire_ranks):.0f}"
          f"  post_kl={np.mean(r.post_fire_kl):.4f}  surr_kl={np.mean(r.surrounding_kl):.4f}"
          f"  glob_kl={np.mean(r.global_kl):.4f}")

# %% blast-radius-plot
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
cmap = plt.colormaps["viridis"]
colors = [cmap(i / (len(n_examples_list) - 1)) for i in range(len(n_examples_list))]

metrics = [
    (axes[0, 0], "on_fire_p", np.mean, "On-fire: P('o')", "P('o')"),
    (axes[0, 1], "on_fire_ranks", np.median, "On-fire: median rank('o')", "Median rank"),
    (axes[0, 2], "post_fire_kl", np.mean, "Post-fire: KL (+1..+5, excl. firings)", "KL"),
    (axes[1, 0], "surrounding_kl", np.mean, "Surrounding: KL (all non-fire in window)", "KL"),
    (axes[1, 1], "global_kl", np.mean, "Global: KL (random text)", "KL"),
]

for ax, attr, agg, title, ylabel in metrics:
    for i, n_ex in enumerate(n_examples_list):
        ys = [agg(getattr(results_grid[(n_ex, s)], attr)) for s in scales_br]
        ax.plot(scales_br, ys, color=colors[i], linewidth=1.8, label=f"n={n_ex}")
    ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.3)
    ax.set_xlabel("U delta scale")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)

# Bottom-right: overlay 3 KL regimes for n=8
ax_ov = axes[1, 2]
n_show = 8
for attr, label, color, ls in [
    ("post_fire_kl", "post-fire (+1..+5)", "#f85149", "-"),
    ("surrounding_kl", "surrounding (whole window)", "#f0883e", "--"),
    ("global_kl", "global (random text)", "#8b949e", ":"),
]:
    ys = [np.mean(getattr(results_grid[(n_show, s)], attr)) for s in scales_br]
    ax_ov.plot(scales_br, ys, color=color, linewidth=2, linestyle=ls, label=label)
ax_ov.axvline(x=1.0, color='gray', linestyle='--', alpha=0.3)
ax_ov.set_xlabel("U delta scale")
ax_ov.set_ylabel("KL")
ax_ov.set_title(f"KL comparison (n={n_show})")
ax_ov.legend(fontsize=8)

fig.suptitle(":2359 edit blast radius — U scale × n training examples", fontsize=13, y=1.01)
fig.tight_layout()
plt.show()

# Summary table
print("\nAt scale=1.0:")
print(f"{'n':>3s}  {'P(o)':>6s}  {'rank':>5s}  {'post':>7s}  {'surr':>7s}  {'glob':>7s}  {'surr/glob':>9s}")
print("-" * 55)
for n_ex in n_examples_list:
    r = results_grid[(n_ex, 1.0)]
    sg = np.mean(r.surrounding_kl) / np.mean(r.global_kl) if np.mean(r.global_kl) > 1e-6 else float('inf')
    print(f"{n_ex:>3d}  {np.mean(r.on_fire_p):>5.1%}  {np.median(r.on_fire_ranks):>5.0f}"
          f"  {np.mean(r.post_fire_kl):>7.4f}  {np.mean(r.surrounding_kl):>7.4f}"
          f"  {np.mean(r.global_kl):>7.4f}  {sg:>8.1f}x")

# %% md-geometry
# ## 9. Geometry of trained write vectors
#
# How do the learned U rows relate to each other and to the target token's unembed
# vector as a function of n training examples?

# %% geometry
# Re-train for each n to extract the actual U vectors and deltas
target_unembed = em_orig.model.target_model.lm_head.weight[TARGET_TOKEN].detach().float()
target_unembed_normed = target_unembed / target_unembed.norm()

firings_2359 = get_firing_positions(key_2359)
random.seed(42)
random.shuffle(firings_2359)

geo_data = {}  # n_ex -> {"u": trained U row, "delta": U_trained - U_original}

for n_ex in n_examples_list:
    train_seqs = make_train_seqs(firings_2359[:n_ex])
    em_g, _ = EditableModel.from_wandb(JOSE)
    tr_g = ComponentTrainer(em_g.model, targets={key_2359: "write"}, lr=1e-3)
    for _ in range(100):
        for tokens_mut, positions in train_seqs:
            logits = tr_g(tokens_mut.unsqueeze(0))
            pos_t = torch.tensor(positions, device=tokens_mut.device)
            loss = torch.nn.functional.cross_entropy(logits[0, positions], tokens_mut[pos_t + 1])
            tr_g.step(loss)
    tr_g.cleanup()

    u_t = em_g.model.components[module_2359].U[cidx_2359].detach().float()
    geo_data[n_ex] = {"u": u_t.clone(), "delta": (u_t - u_original).clone()}
    del em_g

# Norms
print("=== U vector norms ===\n")
print(f"  Original U norm: {u_original.norm().item():.4f}")
print(f"  {'n':>3s}  {'|U|':>7s}  {'|delta|':>8s}  {'|U|/|U_orig|':>12s}")
print("-" * 40)
for n_ex in n_examples_list:
    u = geo_data[n_ex]["u"]
    d = geo_data[n_ex]["delta"]
    print(f"  {n_ex:>3d}  {u.norm().item():>7.4f}  {d.norm().item():>8.4f}  {u.norm().item()/u_original.norm().item():>11.1f}x")

# Cosine with unembed (negated, since double-negation mechanism)
print("\n=== Alignment with unembed('o') [negated cosine — higher = more anti-aligned] ===\n")
print(f"  {'n':>3s}  {'-cos(delta, emb)':>17s}  {'-cos(U, emb)':>13s}")
print("-" * 40)
for n_ex in n_examples_list:
    d_normed = geo_data[n_ex]["delta"] / geo_data[n_ex]["delta"].norm()
    u_normed = geo_data[n_ex]["u"] / geo_data[n_ex]["u"].norm()
    cos_d = -(d_normed @ target_unembed_normed).item()
    cos_u = -(u_normed @ target_unembed_normed).item()
    print(f"  {n_ex:>3d}  {cos_d:>17.4f}  {cos_u:>13.4f}")

# %% geometry-pairwise
# Pairwise cosine similarity matrices: deltas, U vectors, with unembed appended

labels = [f"n={n}" for n in n_examples_list] + ["unembed('o')"]

# Build vector lists
delta_vecs_geo = [-geo_data[n]["delta"] for n in n_examples_list] + [target_unembed]
u_vecs_geo = [-geo_data[n]["u"] for n in n_examples_list] + [target_unembed]

def pairwise_cos(vecs: list[Tensor]) -> np.ndarray:
    normed = torch.stack([v / v.norm() for v in vecs])
    return (normed @ normed.T).cpu().numpy()

cos_deltas = pairwise_cos(delta_vecs_geo)
cos_us = pairwise_cos(u_vecs_geo)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

for ax, matrix, title in [
    (ax1, cos_deltas, "Pairwise cosine: deltas (+ unembed)"),
    (ax2, cos_us, "Pairwise cosine: trained U (+ unembed)"),
]:
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_title(title)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(matrix[i, j]) > 0.6 else "black")
    plt.colorbar(im, ax=ax, shrink=0.8)

fig.tight_layout()
plt.show()

# %% geometry-graphs
ns = np.array(n_examples_list)

u_norms = [geo_data[n]["u"].norm().item() for n in n_examples_list]
delta_norms = [geo_data[n]["delta"].norm().item() for n in n_examples_list]
neg_cos_delta_emb = [-(geo_data[n]["delta"] / geo_data[n]["delta"].norm() @ target_unembed_normed).item() for n in n_examples_list]
neg_cos_u_emb = [-(geo_data[n]["u"] / geo_data[n]["u"].norm() @ target_unembed_normed).item() for n in n_examples_list]

# Pairwise cos between consecutive n-values (delta)
pw_delta = []
for i in range(len(n_examples_list) - 1):
    d1 = geo_data[n_examples_list[i]]["delta"]
    d2 = geo_data[n_examples_list[i + 1]]["delta"]
    pw_delta.append(torch.nn.functional.cosine_similarity(d1, d2, dim=0).item())

fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))

# 1: Norms
axes[0].plot(ns, u_norms, "o-", color="#58a6ff", label="|U trained|")
axes[0].plot(ns, delta_norms, "s--", color="#f0883e", label="|delta|")
axes[0].axhline(u_original.norm().item(), color="gray", linestyle=":", alpha=0.5, label="|U original|")
axes[0].set_xlabel("n training examples")
axes[0].set_ylabel("Norm")
axes[0].set_xscale("log", base=2)
axes[0].set_xticks(ns)
axes[0].set_xticklabels(ns)
axes[0].set_title("U / delta norms")
axes[0].legend(fontsize=8)

# 2: Negated cosine with unembed
axes[1].plot(ns, neg_cos_delta_emb, "o-", color="#58a6ff", label="-cos(delta, emb)")
axes[1].plot(ns, neg_cos_u_emb, "s--", color="#f0883e", label="-cos(U, emb)")
axes[1].set_xlabel("n training examples")
axes[1].set_ylabel("-cos(·, unembed('o'))")
axes[1].set_xscale("log", base=2)
axes[1].set_xticks(ns)
axes[1].set_xticklabels(ns)
axes[1].set_title("Alignment with unembed('o')")
axes[1].set_ylim(0, 1)
axes[1].legend(fontsize=8)

# 3: Pairwise cos(delta_n, delta_{n+1})
axes[2].plot(ns[1:], pw_delta, "o-", color="#58a6ff")
axes[2].set_xlabel("n training examples")
axes[2].set_ylabel("cos(delta_n, delta_{n-1})")
axes[2].set_xscale("log", base=2)
axes[2].set_xticks(ns[1:])
axes[2].set_xticklabels(ns[1:])
axes[2].set_title("Adjacent delta similarity")
axes[2].set_ylim(0.7, 1.0)

# 4: cos(delta_n, delta_n=1) — all vs smallest
d1 = geo_data[1]["delta"]
cos_vs_n1 = [torch.nn.functional.cosine_similarity(d1, geo_data[n]["delta"], dim=0).item() for n in n_examples_list]
axes[3].plot(ns, cos_vs_n1, "o-", color="#58a6ff")
axes[3].set_xlabel("n training examples")
axes[3].set_ylabel("cos(delta_n, delta_1)")
axes[3].set_xscale("log", base=2)
axes[3].set_xticks(ns)
axes[3].set_xticklabels(ns)
axes[3].set_title("Delta similarity vs n=1 baseline")
axes[3].set_ylim(0.7, 1.0)

fig.suptitle(":2359 write-vector geometry vs n training examples", fontsize=13, y=1.02)
fig.tight_layout()
plt.show()

# %% md-analytical
# ## 10. Analytical replacement: set U = -unembed('o')
#
# Instead of training, just replace the U row with the negated unembed vector for 'o'
# (negated because of double-negation mechanism). Sweep the norm to find the right strength.

# %% analytical
firings_2359 = get_firing_positions(key_2359)
random.seed(42)
random.shuffle(firings_2359)

eval_tokens_a, eval_positions_a, eval_fire_sets_a = make_eval_seqs(firings_2359[32:82])

# The analytical delta: -unembed('o'), normalized, then scaled
unembed_dir = -target_unembed_normed  # anti-aligned direction

em_a, _ = EditableModel.from_wandb(JOSE)
em_a.model.components[module_2359].U.data[cidx_2359] = u_original.clone()
tr_a = ComponentTrainer(em_a.model, targets={key_2359: "write"}, lr=1e-3)
baselines_a = cache_baselines(tr_a, eval_tokens_a, global_tokens)

# Sweep norm of the analytical replacement
# For comparison: trained n=8 delta has norm ~2.34 and -cos ~0.64 with unembed
norms_to_try = np.linspace(0, 6, 25)
analytical_results = []

for norm_val in norms_to_try:
    em_a.model.components[module_2359].U.data[cidx_2359] = u_original + norm_val * unembed_dir
    br = measure_blast_radius(tr_a, baselines_a, eval_tokens_a, eval_positions_a, eval_fire_sets_a, global_tokens)
    analytical_results.append((norm_val, br))

tr_a.cleanup()

# Also get the trained n=8 result for comparison on the same eval set
em_t8, _ = EditableModel.from_wandb(JOSE)
tr_t8 = ComponentTrainer(em_t8.model, targets={key_2359: "write"}, lr=1e-3)
baselines_t8 = cache_baselines(tr_t8, eval_tokens_a, global_tokens)

train_seqs_t8 = make_train_seqs(firings_2359[:8])
for _ in range(100):
    for tokens_mut, positions in train_seqs_t8:
        logits = tr_t8(tokens_mut.unsqueeze(0))
        pos_t = torch.tensor(positions, device=tokens_mut.device)
        loss = torch.nn.functional.cross_entropy(logits[0, positions], tokens_mut[pos_t + 1])
        tr_t8.step(loss)

br_trained = measure_blast_radius(tr_t8, baselines_t8, eval_tokens_a, eval_positions_a, eval_fire_sets_a, global_tokens)
tr_t8.cleanup()
del em_t8

print(f"Trained n=8: P('o')={np.mean(br_trained.on_fire_p):.1%}  "
      f"post_kl={np.mean(br_trained.post_fire_kl):.4f}  glob_kl={np.mean(br_trained.global_kl):.4f}")

# %% analytical-plot
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

norms_arr = [d[0] for d in analytical_results]
p_on = [np.mean(d[1].on_fire_p) for d in analytical_results]
rank_on = [np.median(d[1].on_fire_ranks) for d in analytical_results]
kl_surr = [np.mean(d[1].surrounding_kl) for d in analytical_results]
kl_glob = [np.mean(d[1].global_kl) for d in analytical_results]

# Trained n=8 reference values
ref_p = np.mean(br_trained.on_fire_p)
ref_surr = np.mean(br_trained.surrounding_kl)
ref_glob = np.mean(br_trained.global_kl)

axes[0].plot(norms_arr, p_on, "o-", color="#58a6ff", markersize=4, label="analytical -unembed")
axes[0].axhline(ref_p, color="#f0883e", linestyle="--", label=f"trained n=8 ({ref_p:.0%})")
axes[0].set_xlabel("|delta| (norm of -unembed direction)")
axes[0].set_ylabel("P('o') on-fire")
axes[0].set_title("On-target effect")
axes[0].legend(fontsize=8)

axes[1].plot(norms_arr, rank_on, "o-", color="#58a6ff", markersize=4)
axes[1].axhline(np.median(br_trained.on_fire_ranks), color="#f0883e", linestyle="--", label="trained n=8")
axes[1].set_xlabel("|delta|")
axes[1].set_ylabel("Median rank('o')")
axes[1].set_title("On-target rank")
axes[1].legend(fontsize=8)

axes[2].plot(norms_arr, kl_surr, "o-", color="#f85149", markersize=4, label="surrounding KL (analytical)")
axes[2].plot(norms_arr, kl_glob, "s-", color="#8b949e", markersize=4, label="global KL (analytical)")
axes[2].axhline(ref_surr, color="#f85149", linestyle="--", alpha=0.5, label="surr KL trained n=8")
axes[2].axhline(ref_glob, color="#8b949e", linestyle="--", alpha=0.5, label="glob KL trained n=8")
axes[2].set_xlabel("|delta|")
axes[2].set_ylabel("KL")
axes[2].set_title("Off-target damage")
axes[2].legend(fontsize=8)

fig.suptitle("Analytical U = -unembed('o') vs trained write vector", fontsize=13, y=1.02)
fig.tight_layout()
plt.show()

# Find the norm that matches trained n=8 P('o')
best_match = min(analytical_results, key=lambda d: abs(np.mean(d[1].on_fire_p) - ref_p))
br_match = best_match[1]
print(f"\nNorm matching trained P('o') ({ref_p:.1%}): |delta|={best_match[0]:.2f}")
print(f"  Analytical:  P('o')={np.mean(br_match.on_fire_p):.1%}  surr_kl={np.mean(br_match.surrounding_kl):.4f}  glob_kl={np.mean(br_match.global_kl):.4f}")
print(f"  Trained n=8: P('o')={ref_p:.1%}  surr_kl={ref_surr:.4f}  glob_kl={ref_glob:.4f}")

# %% md-emoji-mass
# ## 11. Baseline emoji probability mass
#
# When :2359 fires, how much probability mass does the model originally assign to
# emoji-like completions? This gives us a ceiling: if the edit just redirects all
# emoji probability to 'o', P('o') can't exceed this.

# %% emoji-mass
emoticon_chars = set(")(DPpSs/\\|oO3><*@#^-]xXFf:;=")  # mouths, noses, eyes

firings_2359 = get_firing_positions(key_2359)
random.seed(42)
random.shuffle(firings_2359)

# Use a fresh model for baseline (no edits)
em_base, _ = EditableModel.from_wandb(JOSE)
tr_base = ComponentTrainer(em_base.model, targets={key_2359: "write"}, lr=1e-3)

# Measure baseline emoji mass at all firing positions
emoji_token_ids = set()
for tid in range(tok.vocab_size):
    s = tok.get_tok_display(tid).strip()
    if len(s) >= 1 and all(c in emoticon_chars for c in s):
        emoji_token_ids.add(tid)

print(f"{len(emoji_token_ids)} tokens classified as emoji-like characters")
print("Examples:", [tok.get_tok_display(t) for t in sorted(emoji_token_ids)[:20]])

emoji_masses = []
p_o_baseline = []

with torch.no_grad():
    for ex, p in firings_2359[:200]:
        tokens_t = torch.tensor(ex.token_ids, device="cuda")
        probs = tr_base(tokens_t.unsqueeze(0))[0, p].softmax(-1)
        emoji_mass = sum(probs[tid].item() for tid in emoji_token_ids)
        emoji_masses.append(emoji_mass)
        p_o_baseline.append(probs[TARGET_TOKEN].item())

tr_base.cleanup()
del em_base

mean_emoji_mass = np.mean(emoji_masses)
mean_p_o_base = np.mean(p_o_baseline)

print(f"\nAt firing positions (n={len(emoji_masses)}):")
print(f"  Mean emoji-char probability mass: {mean_emoji_mass:.1%}")
print(f"  Median: {np.median(emoji_masses):.1%}")
print(f"  Baseline P('o'): {mean_p_o_base:.1%}")

# %% emoji-mass-plot
# Re-plot the analytical curve with the emoji mass ceiling marked

fig, ax = plt.subplots(figsize=(10, 5))

norms_arr = [d[0] for d in analytical_results]
p_on = [np.mean(d[1].on_fire_p) for d in analytical_results]

ax.plot(norms_arr, p_on, "o-", color="#58a6ff", markersize=4, label="analytical -unembed('o')")
ax.axhline(ref_p, color="#f0883e", linestyle="--", label=f"trained n=8 ({ref_p:.0%})")
ax.axhline(mean_emoji_mass, color="#7ee787", linestyle="-.", linewidth=2,
           label=f"baseline emoji prob mass ({mean_emoji_mass:.0%})")
ax.axhline(mean_p_o_base, color="#8b949e", linestyle=":", alpha=0.5,
           label=f"baseline P('o') ({mean_p_o_base:.1%})")

ax.set_xlabel("|delta| (norm of -unembed direction)")
ax.set_ylabel("P('o') at firing positions")
ax.set_title(":2359 — P('o') vs edit strength, with emoji mass ceiling")
ax.set_ylim(-0.02, 1.02)
ax.legend(fontsize=9)
fig.tight_layout()
plt.show()

print(f"\nThe edit pushes P('o') well beyond the baseline emoji mass ({mean_emoji_mass:.0%}),")
print("meaning it's not just redirecting emoji probability — it's pulling mass from non-emoji tokens too.")

# %% md-summary
# ## Summary

# %% summary
fig = plt.figure(figsize=(20, 16))
gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.35)

# --- Row 1: Core result ---

# 1a: P('o') vs n_examples at scale=1
ax = fig.add_subplot(gs[0, 0])
ns_plot = n_examples_list
p_by_n = [np.mean(results_grid[(n, 1.0)].on_fire_p) for n in ns_plot]
ax.plot(ns_plot, p_by_n, "o-", color="#58a6ff", linewidth=2)
ax.axhline(mean_emoji_mass, color="#7ee787", linestyle="-.", alpha=0.7, label=f"emoji mass ({mean_emoji_mass:.0%})")
ax.set_xscale("log", base=2)
ax.set_xticks(ns_plot)
ax.set_xticklabels(ns_plot)
ax.set_xlabel("n training examples")
ax.set_ylabel("P('o')")
ax.set_title("On-target accuracy")
ax.set_ylim(0, 1)
ax.legend(fontsize=7)

# 1b: Off-target KL (3 regimes) vs n_examples at scale=1
ax = fig.add_subplot(gs[0, 1])
for attr, label, color in [
    ("post_fire_kl", "post-fire", "#f85149"),
    ("surrounding_kl", "surrounding", "#f0883e"),
    ("global_kl", "global", "#8b949e"),
]:
    vals = [np.mean(getattr(results_grid[(n, 1.0)], attr)) for n in ns_plot]
    ax.plot(ns_plot, vals, "o-", color=color, linewidth=1.5, label=label)
ax.set_xscale("log", base=2)
ax.set_xticks(ns_plot)
ax.set_xticklabels(ns_plot)
ax.set_xlabel("n training examples")
ax.set_ylabel("Mean KL")
ax.set_title("Off-target damage")
ax.legend(fontsize=7)

# 1c: Blast radius — P('o') vs scale for n=1,4,8,32
ax = fig.add_subplot(gs[0, 2])
for n_ex, color in [(1, "#8b949e"), (4, "#58a6ff"), (8, "#7ee787"), (32, "#f0883e")]:
    ys = [np.mean(results_grid[(n_ex, s)].on_fire_p) for s in scales_br]
    ax.plot(scales_br, ys, color=color, linewidth=1.5, label=f"n={n_ex}")
ax.axhline(mean_emoji_mass, color="#7ee787", linestyle="-.", alpha=0.3)
ax.axvline(1.0, color="gray", linestyle="--", alpha=0.3)
ax.set_xlabel("U delta scale")
ax.set_ylabel("P('o')")
ax.set_title("P('o') vs edit strength")
ax.legend(fontsize=7)

# 1d: Blast radius — surrounding KL vs scale
ax = fig.add_subplot(gs[0, 3])
for n_ex, color in [(1, "#8b949e"), (4, "#58a6ff"), (8, "#7ee787"), (32, "#f0883e")]:
    ys = [np.mean(results_grid[(n_ex, s)].surrounding_kl) for s in scales_br]
    ax.plot(scales_br, ys, color=color, linewidth=1.5, label=f"n={n_ex}")
ax.axvline(1.0, color="gray", linestyle="--", alpha=0.3)
ax.set_xlabel("U delta scale")
ax.set_ylabel("Surrounding KL")
ax.set_title("Surrounding KL vs edit strength")
ax.legend(fontsize=7)

# --- Row 2: Geometry ---

# 2a: Delta norms vs n
ax = fig.add_subplot(gs[1, 0])
delta_norms_plot = [geo_data[n]["delta"].norm().item() for n in ns_plot]
ax.plot(ns_plot, delta_norms_plot, "o-", color="#58a6ff", linewidth=2)
ax.axhline(u_original.norm().item(), color="gray", linestyle=":", alpha=0.5, label="|U original|")
ax.set_xscale("log", base=2)
ax.set_xticks(ns_plot)
ax.set_xticklabels(ns_plot)
ax.set_xlabel("n training examples")
ax.set_ylabel("|delta|")
ax.set_title("Learned delta norm")
ax.legend(fontsize=7)

# 2b: Unembed alignment vs n
ax = fig.add_subplot(gs[1, 1])
neg_cos = [-(geo_data[n]["delta"] / geo_data[n]["delta"].norm() @ target_unembed_normed).item() for n in ns_plot]
ax.plot(ns_plot, neg_cos, "o-", color="#58a6ff", linewidth=2)
ax.set_xscale("log", base=2)
ax.set_xticks(ns_plot)
ax.set_xticklabels(ns_plot)
ax.set_xlabel("n training examples")
ax.set_ylabel("-cos(delta, unembed('o'))")
ax.set_title("Unembed alignment")
ax.set_ylim(0, 1)

# 2c: Pairwise cosine of deltas (heatmap)
ax = fig.add_subplot(gs[1, 2])
cos_d = pairwise_cos(delta_vecs_geo)
im = ax.imshow(cos_d, cmap="RdBu_r", vmin=-1, vmax=1)
labels_pw = [f"n={n}" for n in ns_plot] + ["emb"]
ax.set_xticks(range(len(labels_pw)))
ax.set_yticks(range(len(labels_pw)))
ax.set_xticklabels(labels_pw, fontsize=8, rotation=45, ha="right")
ax.set_yticklabels(labels_pw, fontsize=8)
for i in range(len(labels_pw)):
    for j in range(len(labels_pw)):
        ax.text(j, i, f"{cos_d[i,j]:.2f}", ha="center", va="center", fontsize=7,
                color="white" if abs(cos_d[i,j]) > 0.6 else "black")
ax.set_title("Pairwise cos(-delta, unembed)")
plt.colorbar(im, ax=ax, shrink=0.8)

# 2d: Single-example convergence histogram
ax = fig.add_subplot(gs[1, 3])
ax.hist(upper_tri, bins=15, color="#58a6ff", edgecolor="white", alpha=0.8)
ax.axvline(np.mean(upper_tri), color="#f0883e", linestyle="--", label=f"mean={np.mean(upper_tri):.2f}")
ax.set_xlabel("Pairwise cosine similarity")
ax.set_ylabel("Count")
ax.set_title(f"Single-example delta convergence (N={N_RUNS})")
ax.legend(fontsize=7)

# --- Row 3: Analytical replacement ---

# 3a: Analytical P('o') vs norm with emoji ceiling
ax = fig.add_subplot(gs[2, 0:2])
norms_arr = [d[0] for d in analytical_results]
p_analytical = [np.mean(d[1].on_fire_p) for d in analytical_results]
ax.plot(norms_arr, p_analytical, "o-", color="#58a6ff", markersize=4, linewidth=2, label="analytical: U = -unembed('o')")
ax.axhline(ref_p, color="#f0883e", linestyle="--", linewidth=1.5, label=f"trained n=8 ({ref_p:.0%})")
ax.axhline(mean_emoji_mass, color="#7ee787", linestyle="-.", linewidth=1.5, label=f"baseline emoji mass ({mean_emoji_mass:.0%})")
# Mark the matching point
ax.plot(best_match[0], np.mean(best_match[1].on_fire_p), "s", color="#f0883e", markersize=10, zorder=5)
ax.annotate(f"|delta|={best_match[0]:.1f}", (best_match[0], np.mean(best_match[1].on_fire_p)),
            textcoords="offset points", xytext=(10, -15), fontsize=9, color="#f0883e")
ax.set_xlabel("|delta| (norm of -unembed direction)")
ax.set_ylabel("P('o')")
ax.set_title("Analytical write-vector replacement: no training needed")
ax.set_ylim(-0.02, 1.02)
ax.legend(fontsize=9)

# 3b: Analytical off-target
ax = fig.add_subplot(gs[2, 2:4])
kl_surr_a = [np.mean(d[1].surrounding_kl) for d in analytical_results]
kl_glob_a = [np.mean(d[1].global_kl) for d in analytical_results]
ax.plot(norms_arr, kl_surr_a, "o-", color="#f85149", markersize=4, linewidth=1.5, label="surrounding KL (analytical)")
ax.plot(norms_arr, kl_glob_a, "s-", color="#8b949e", markersize=4, linewidth=1.5, label="global KL (analytical)")
ax.axhline(ref_surr, color="#f85149", linestyle="--", alpha=0.5, label=f"surr KL trained n=8 ({ref_surr:.4f})")
ax.axhline(ref_glob, color="#8b949e", linestyle="--", alpha=0.5, label=f"glob KL trained n=8 ({ref_glob:.4f})")
ax.plot(best_match[0], np.mean(best_match[1].surrounding_kl), "s", color="#f85149", markersize=10, zorder=5)
ax.set_xlabel("|delta| (norm of -unembed direction)")
ax.set_ylabel("KL")
ax.set_title("Analytical replacement: off-target damage")
ax.legend(fontsize=8)

fig.suptitle("Write-vector editing of SPD component :2359 (emoticon → 'o')", fontsize=15, y=1.01)
plt.show()

# --- Text summary ---
print("""
KEY FINDINGS
============

1. WRITE-ONLY EDITING WORKS
   Training a single component's U row (write vector) redirects its output token
   with high accuracy and minimal collateral damage:
     n=1:  73% P('o'),  surr KL = 0.006,  glob KL = 0.005
     n=8:  85% P('o'),  surr KL = 0.011,  glob KL = 0.006
     n=32: 97% P('o'),  surr KL = 0.184,  glob KL = 0.033

2. THE EDIT IS HIGHLY LOCALIZED
   KL is concentrated at the firing position. Post-fire and surrounding tokens
   show KL barely above the global baseline (1.3-1.8x for n≤8).

3. SINGLE EXAMPLES LEARN A CONSISTENT DIRECTION
   10 single-example runs produce deltas with mean pairwise cosine = 0.63.
   The direction is not memorized — it's a property of the component geometry.

4. THE DELTA IS ANTI-ALIGNED WITH THE TARGET UNEMBED
   cos(delta, unembed('o')) = -0.50 (n=1) to -0.66 (n=16).
   Double-negation: negative activation × anti-aligned write → positive logit.

5. NO TRAINING NEEDED: ANALYTICAL REPLACEMENT WORKS
   Setting U = U_orig - 3.0 * unembed('o')/|unembed('o')| matches trained n=8
   performance (86% vs 85% P('o'), 0.013 vs 0.011 surrounding KL).

6. THE EDIT EXCEEDS THE EMOJI PROBABILITY CEILING
   Baseline emoji probability mass at firing positions: 51%.
   The edit pushes P('o') to 85-95% — pulling mass from non-emoji tokens too.
""")

# %% md-direct-assign
# ## 12. Direct assignment: U = -unembed('o') (no original U)
#
# Previous analytical experiment added -unembed as a delta on top of U_original.
# Now try just assigning U_row = scale * (-unembed), ignoring the original U entirely.

# %% direct-assign
em_da, _ = EditableModel.from_wandb(JOSE)
em_da.model.components[module_2359].U.data[cidx_2359] = u_original.clone()
tr_da = ComponentTrainer(em_da.model, targets={key_2359: "write"}, lr=1e-3)
baselines_da = cache_baselines(tr_da, eval_tokens_a, global_tokens)

norms_da = np.linspace(0, 6, 25)
direct_results = []

for norm_val in norms_da:
    # Direct assignment: U = norm * (-unembed), NOT U_orig + delta
    em_da.model.components[module_2359].U.data[cidx_2359] = (norm_val * (-target_unembed_normed)).to(u_original.dtype)
    br = measure_blast_radius(tr_da, baselines_da, eval_tokens_a, eval_positions_a, eval_fire_sets_a, global_tokens)
    direct_results.append((norm_val, br))

tr_da.cleanup()

# %% direct-assign-plot
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

norms_direct = [d[0] for d in direct_results]
p_direct = [np.mean(d[1].on_fire_p) for d in direct_results]
norms_add = [d[0] for d in analytical_results]
p_add = [np.mean(d[1].on_fire_p) for d in analytical_results]

# P('o') comparison
axes[0].plot(norms_direct, p_direct, "o-", color="#58a6ff", markersize=4, linewidth=2, label="direct: U = α·(-emb)")
axes[0].plot(norms_add, p_add, "s--", color="#f0883e", markersize=4, linewidth=1.5, label="additive: U = U_orig + α·(-emb)")
axes[0].axhline(ref_p, color="gray", linestyle=":", label=f"trained n=8 ({ref_p:.0%})")
axes[0].axhline(mean_emoji_mass, color="#7ee787", linestyle="-.", alpha=0.5, label=f"emoji mass ({mean_emoji_mass:.0%})")
axes[0].set_xlabel("α (norm)")
axes[0].set_ylabel("P('o')")
axes[0].set_title("On-target: P('o')")
axes[0].set_ylim(-0.02, 1.02)
axes[0].legend(fontsize=7)

# Surrounding KL
kl_surr_direct = [np.mean(d[1].surrounding_kl) for d in direct_results]
kl_surr_add = [np.mean(d[1].surrounding_kl) for d in analytical_results]
axes[1].plot(norms_direct, kl_surr_direct, "o-", color="#58a6ff", markersize=4, linewidth=2, label="direct")
axes[1].plot(norms_add, kl_surr_add, "s--", color="#f0883e", markersize=4, linewidth=1.5, label="additive")
axes[1].axhline(ref_surr, color="gray", linestyle=":", label="trained n=8")
axes[1].set_xlabel("α")
axes[1].set_ylabel("Surrounding KL")
axes[1].set_title("Off-target: surrounding KL")
axes[1].legend(fontsize=7)

# Global KL
kl_glob_direct = [np.mean(d[1].global_kl) for d in direct_results]
kl_glob_add = [np.mean(d[1].global_kl) for d in analytical_results]
axes[2].plot(norms_direct, kl_glob_direct, "o-", color="#58a6ff", markersize=4, linewidth=2, label="direct")
axes[2].plot(norms_add, kl_glob_add, "s--", color="#f0883e", markersize=4, linewidth=1.5, label="additive")
axes[2].axhline(ref_glob, color="gray", linestyle=":", label="trained n=8")
axes[2].set_xlabel("α")
axes[2].set_ylabel("Global KL")
axes[2].set_title("Off-target: global KL")
axes[2].legend(fontsize=7)

fig.suptitle("Direct U assignment vs additive delta (both analytical, no training)", fontsize=13, y=1.02)
fig.tight_layout()
plt.show()

# Find norm matching trained P('o') for direct
best_direct = min(direct_results, key=lambda d: abs(np.mean(d[1].on_fire_p) - ref_p))
print(f"Direct assignment matching trained P('o') ({ref_p:.1%}): α={best_direct[0]:.2f}")
print(f"  Direct:   P('o')={np.mean(best_direct[1].on_fire_p):.1%}  surr_kl={np.mean(best_direct[1].surrounding_kl):.4f}  glob_kl={np.mean(best_direct[1].global_kl):.4f}")
print(f"  Additive: P('o')={np.mean(best_match[1].on_fire_p):.1%}  surr_kl={np.mean(best_match[1].surrounding_kl):.4f}  glob_kl={np.mean(best_match[1].global_kl):.4f}")
print(f"  Trained:  P('o')={ref_p:.1%}  surr_kl={ref_surr:.4f}  glob_kl={ref_glob:.4f}")


# %% md-lora
# ## 13. LoRA baseline comparison
#
# Train a rank-1 LoRA on the same layer's full weight matrix to predict 'o' at
# the same firing positions. Compare on-target and off-target metrics to see
# whether SPD component editing offers any advantage over naive LoRA.

# %% lora
# Find the actual nn.Linear module for h.2.mlp.down_proj
target_module_path = "h.2.mlp.down_proj"
parts = target_module_path.split(".")
mod = em.model.target_model
for part in parts:
    mod = getattr(mod, part)
target_linear = mod

W_shape = target_linear.weight.shape  # [d_out, d_in]
print(f"Target layer: {target_module_path}, weight shape: {W_shape}")

firings_2359 = get_firing_positions(key_2359)
random.seed(42)
random.shuffle(firings_2359)

# Same eval sets as blast radius
eval_tokens_l, eval_positions_l, eval_fire_sets_l = make_eval_seqs(firings_2359[32:82])


@dataclass
class LoRAAdapter:
    """Rank-1 LoRA: delta_W = B @ A where A is [rank, d_in], B is [d_out, rank]."""
    A: Tensor  # [1, d_in]
    B: Tensor  # [d_out, 1]
    hook_handle: torch.utils.hooks.RemovableHandle | None = None

    def install(self, linear: torch.nn.Linear) -> None:
        def hook(_mod: torch.nn.Module, _input: tuple, output: Tensor) -> Tensor:
            # LoRA adds B @ A @ x to the output
            # output is [..., d_out], input[0] is [..., d_in]
            x = _input[0]
            lora_out = (x @ self.A.T) @ self.B.T  # [..., 1] @ [1, d_out] = [..., d_out]
            return output + lora_out
        self.hook_handle = linear.register_forward_hook(hook)

    def remove(self) -> None:
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None

    def parameters(self) -> list[Tensor]:
        return [self.A, self.B]


n_examples_lora = [1, 2, 4, 8, 16, 32]
lora_results: dict[int, BlastRadiusResult] = {}

for n_ex in n_examples_lora:
    train_seqs_l = make_train_seqs(firings_2359[:n_ex])

    # Fresh target model (no SPD, just the original LM)
    em_l, _ = EditableModel.from_wandb(JOSE)
    target_model = em_l.model.target_model
    target_model.eval()

    # Find the linear
    mod_l = target_model
    for part in target_module_path.split("."):
        mod_l = getattr(mod_l, part)

    # Init LoRA rank-1
    d_out, d_in = mod_l.weight.shape
    lora = LoRAAdapter(
        A=torch.randn(1, d_in, device="cuda") * 0.01,
        B=torch.zeros(d_out, 1, device="cuda"),
    )
    lora.A.requires_grad = True
    lora.B.requires_grad = True
    lora.install(mod_l)

    optimizer = torch.optim.AdamW([lora.A, lora.B], lr=1e-3)

    # Cache baselines (with LoRA installed but at init = ~zero effect)
    baselines_l = {}
    with torch.no_grad():
        for tokens_t in list(eval_tokens_l) + global_tokens:
            baselines_l[id(tokens_t)] = target_model(tokens_t.unsqueeze(0)).logits[0].softmax(-1)

    # Train
    for _ in range(100):
        for tokens_mut, positions in train_seqs_l:
            logits = target_model(tokens_mut.unsqueeze(0)).logits
            pos_t = torch.tensor(positions, device="cuda")
            loss = torch.nn.functional.cross_entropy(logits[0, positions], tokens_mut[pos_t + 1])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Eval — measure same metrics as blast radius but using raw target_model forward
    r = BlastRadiusResult()
    with torch.no_grad():
        for tokens_t, positions, fire_set in zip(eval_tokens_l, eval_positions_l, eval_fire_sets_l):
            logits = target_model(tokens_t.unsqueeze(0)).logits[0]
            probs = logits.softmax(-1)
            kl = kl_per_token(probs, baselines_l[id(tokens_t)])

            for p in positions:
                r.on_fire_ranks.append((logits[p] >= logits[p, TARGET_TOKEN]).sum().item())
                r.on_fire_p.append(probs[p, TARGET_TOKEN].item())
                for offset in range(1, 6):
                    pos = p + offset
                    if pos < kl.shape[0] and pos not in fire_set:
                        r.post_fire_kl.append(kl[pos].item())
            for i in range(kl.shape[0]):
                if i not in fire_set:
                    r.surrounding_kl.append(kl[i].item())

        for tokens_t in global_tokens:
            probs = target_model(tokens_t.unsqueeze(0)).logits[0].softmax(-1)
            kl = kl_per_token(probs, baselines_l[id(tokens_t)])
            r.global_kl.extend(kl.tolist())

    lora.remove()
    lora_results[n_ex] = r
    del em_l

    print(f"  LoRA n={n_ex:>2d}  P('o')={np.mean(r.on_fire_p):.1%}  "
          f"surr_kl={np.mean(r.surrounding_kl):.4f}  glob_kl={np.mean(r.global_kl):.4f}")

# %% lora-plot
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

ns_arr = np.array(n_examples_lora)

# SPD results from blast radius (scale=1.0)
spd_p = [np.mean(results_grid[(n, 1.0)].on_fire_p) for n in ns_arr]
spd_surr = [np.mean(results_grid[(n, 1.0)].surrounding_kl) for n in ns_arr]
spd_glob = [np.mean(results_grid[(n, 1.0)].global_kl) for n in ns_arr]

lora_p = [np.mean(lora_results[n].on_fire_p) for n in ns_arr]
lora_surr = [np.mean(lora_results[n].surrounding_kl) for n in ns_arr]
lora_glob = [np.mean(lora_results[n].global_kl) for n in ns_arr]

# P('o')
axes[0].plot(ns_arr, spd_p, "o-", color="#58a6ff", linewidth=2, label="SPD write-vector")
axes[0].plot(ns_arr, lora_p, "s--", color="#f0883e", linewidth=2, label="LoRA rank-1")
axes[0].axhline(mean_emoji_mass, color="#7ee787", linestyle="-.", alpha=0.5)
axes[0].set_xscale("log", base=2)
axes[0].set_xticks(ns_arr)
axes[0].set_xticklabels([str(n) for n in ns_arr])
axes[0].set_xlabel("n training examples")
axes[0].set_ylabel("P('o')")
axes[0].set_title("On-target: P('o')")
axes[0].set_ylim(0, 1)
axes[0].legend(fontsize=9)

# Surrounding KL
axes[1].plot(ns_arr, spd_surr, "o-", color="#58a6ff", linewidth=2, label="SPD")
axes[1].plot(ns_arr, lora_surr, "s--", color="#f0883e", linewidth=2, label="LoRA rank-1")
axes[1].set_xscale("log", base=2)
axes[1].set_xticks(ns_arr)
axes[1].set_xticklabels([str(n) for n in ns_arr])
axes[1].set_xlabel("n training examples")
axes[1].set_ylabel("Surrounding KL")
axes[1].set_title("Off-target: surrounding KL")
axes[1].legend(fontsize=9)

# Global KL
axes[2].plot(ns_arr, spd_glob, "o-", color="#58a6ff", linewidth=2, label="SPD")
axes[2].plot(ns_arr, lora_glob, "s--", color="#f0883e", linewidth=2, label="LoRA rank-1")
axes[2].set_xscale("log", base=2)
axes[2].set_xticks(ns_arr)
axes[2].set_xticklabels([str(n) for n in ns_arr])
axes[2].set_xlabel("n training examples")
axes[2].set_ylabel("Global KL")
axes[2].set_title("Off-target: global KL")
axes[2].legend(fontsize=9)

fig.suptitle("SPD component editing vs LoRA rank-1 baseline", fontsize=13, y=1.02)
fig.tight_layout()
plt.show()

# Summary table
print(f"\n{'n':>3s}  {'SPD P(o)':>9s}  {'LoRA P(o)':>10s}  {'SPD surr':>9s}  {'LoRA surr':>10s}  {'SPD glob':>9s}  {'LoRA glob':>10s}")
print("-" * 75)
for n in ns_arr:
    print(f"{n:>3d}  {np.mean(results_grid[(n, 1.0)].on_fire_p):>8.1%}  {np.mean(lora_results[n].on_fire_p):>9.1%}"
          f"  {np.mean(results_grid[(n, 1.0)].surrounding_kl):>9.4f}  {np.mean(lora_results[n].surrounding_kl):>10.4f}"
          f"  {np.mean(results_grid[(n, 1.0)].global_kl):>9.4f}  {np.mean(lora_results[n].global_kl):>10.4f}")

# %% cleanup
trainer.cleanup()
print("Done.")