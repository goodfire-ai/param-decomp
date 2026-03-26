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
from torch import Tensor

from spd.autointerp.repo import InterpRepo
from spd.editing import (
    EditableModel,
    parse_component_key,
    render_edit_comparison,
    search_interpretations,
)
from spd.editing.compare import ExampleDiff, TokenDiff
from spd.editing.component_trainer import ComponentTrainer
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

# %% search_interpretations

asdf = search_interpretations(harvest, interp, "citations")

# %% 
len(asdf)
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

# %% cleanup
trainer.cleanup()
print("Done.")
