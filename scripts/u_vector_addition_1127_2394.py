# %% md-intro
# # U-Vector Addition: Steering Narrative Fiction → All-Caps
#
# Component `h.2.mlp.down_proj:1127` fires on narrative fiction / storytelling contexts.
# Component `h.2.attn.o_proj:899` fires on all-caps text.
#
# We add the U-vector (write direction) of attn o_proj:899 to mlp down_proj:1127's U-vector,
# scaled by a prefactor, to make the model predict all-caps outputs in narrative contexts.
# Both U-vectors write to the same 768-dim residual stream.
#
# Prefactors: 1, -1, 2, -2, 10, -10

# %% setup
import torch
from IPython.display import HTML, display
from jaxtyping import Float, Int
from spd.editing.compare import ExampleDiff, TokenDiff
from spd.editing.component_trainer import ComponentTrainer
from torch import Tensor

from spd.editing import EditableModel, parse_component_key, render_edit_comparison
from spd.harvest.repo import HarvestRepo
from spd.harvest.schemas import ActivationExample

JOSE = "wandb:goodfire/spd/s-55ea3f9b"
JOSE_ID = "s-55ea3f9b"

NARRATIVE_KEY = "h.2.mlp.down_proj:1127"
NARRATIVE_MODULE = "h.2.mlp.down_proj"
NARRATIVE_IDX = 1127

CAPS_KEY = "h.2.attn.o_proj:899"
CAPS_MODULE = "h.2.attn.o_proj"
CAPS_IDX = 899

PREFACTORS = [1, -1, 2, -2, 10, -10]
CONTEXT_RADIUS = 10  # 10 before + firing + 10 after ≈ 20 tokens of context

em, tok = EditableModel.from_wandb(JOSE)
harvest = HarvestRepo.open_most_recent(JOSE_ID)
assert harvest
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


def extract_context_window(
    ex: ActivationExample, fire_pos: int
) -> tuple[list[int], list[bool], int]:
    """Extract a window of tokens around the firing position.

    Returns (token_ids, firings, fire_pos_in_window).
    """
    start = max(0, fire_pos - CONTEXT_RADIUS)
    end = min(len(ex.token_ids), fire_pos + CONTEXT_RADIUS + 1)
    return (
        ex.token_ids[start:end],
        ex.firings[start:end],
        fire_pos - start,
    )


def kl_per_token(
    probs_edit: Float[Tensor, "seq vocab"],
    probs_base: Float[Tensor, "seq vocab"],
) -> Float[Tensor, " seq"]:
    return (probs_edit * ((probs_edit + 1e-10).log() - (probs_base + 1e-10).log())).sum(-1)


def build_diffs(
    trainer: ComponentTrainer,
    baselines: dict[int, Float[Tensor, "seq vocab"]],
    examples: list[tuple[ActivationExample, int]],
    comp_key: str,
) -> list[ExampleDiff]:
    """Build per-token diffs between edited (via trainer) and cached baselines."""
    module, cidx = parse_component_key(comp_key)
    diffs = []
    with torch.no_grad():
        for ex, fire_pos in examples:
            token_ids, firings, fp_in_window = extract_context_window(ex, fire_pos)
            tokens_t = torch.tensor(token_ids, device="cuda")

            probs_edit = trainer(tokens_t.unsqueeze(0))[0].softmax(-1)
            probs_base = baselines[id(tokens_t)]
            kl = kl_per_token(probs_edit, probs_base)

            ci_vals = em.get_ci(tokens_t)[module][:, cidx].cpu()
            act_vals = em.get_component_activations(tokens_t, comp_key).cpu()
            spans = tok.get_spans(token_ids)
            fire_set = {i for i, f in enumerate(firings) if f}

            token_diffs = []
            for t in range(len(spans)):
                diff = probs_edit[t] - probs_base[t]
                inc_idx = diff.topk(5).indices
                dec_idx = (-diff).topk(5).indices
                before_idx = probs_base[t].topk(8).indices
                after_idx = probs_edit[t].topk(8).indices
                token_diffs.append(TokenDiff(
                    span=spans[t],
                    kl=kl[t].item(),
                    ci=ci_vals[t].item(),
                    activation=act_vals[t].item(),
                    fires=t in fire_set,
                    topk_before=[
                        (tok.get_tok_display(int(j)), probs_base[t, j].item()) for j in before_idx
                    ],
                    topk_after=[
                        (tok.get_tok_display(int(j)), probs_edit[t, j].item()) for j in after_idx
                    ],
                    top_increases=[
                        (tok.get_tok_display(int(j)), probs_base[t, j].item(), probs_edit[t, j].item())
                        for j in inc_idx
                    ],
                    top_decreases=[
                        (tok.get_tok_display(int(j)), probs_base[t, j].item(), probs_edit[t, j].item())
                        for j in dec_idx
                    ],
                ))
            diffs.append(ExampleDiff(tokens=token_diffs, max_kl=kl.max().item()))
    diffs.sort(key=lambda d: -d.max_kl)
    return diffs


print("Helpers defined.")

# %% md-components
# ## 1. Inspect the two components

# %% components
firings_narrative = get_firing_positions(NARRATIVE_KEY)
firings_caps = get_firing_positions(CAPS_KEY)
print(f"Narrative :1127 — {len(firings_narrative)} firing positions")
print(f"All-caps  o_proj:899 — {len(firings_caps)} firing positions")

# Show a few example contexts for :1127
print("\nSample narrative firing contexts:")
for ex, fp in firings_narrative[:5]:
    window_ids, _, fp_w = extract_context_window(ex, fp)
    spans = tok.get_spans(window_ids)
    marked = [f"[{s}]" if i == fp_w else s for i, s in enumerate(spans)]
    print(f"  ...{''.join(marked)}...")

# %% md-vectors
# ## 2. U-vector properties

# %% vectors
u_narrative = em.model.components[NARRATIVE_MODULE].U[NARRATIVE_IDX].detach().float()
u_caps = em.model.components[CAPS_MODULE].U[CAPS_IDX].detach().float()

cos = torch.nn.functional.cosine_similarity(u_narrative, u_caps, dim=0).item()
print(f"U(:1127) norm: {u_narrative.norm().item():.4f}")
print(f"U(:2394) norm: {u_caps.norm().item():.4f}")
print(f"Cosine(U_1127, U_899): {cos:.4f}")

boosts_n, supps_n = em.unembed_alignment(NARRATIVE_KEY, tok)
boosts_c, supps_c = em.unembed_alignment(CAPS_KEY, tok)

print("\n:1127 (narrative) top boosted:")
for m in boosts_n[:5]:
    print(f"  {m.token_str:>12s}  cos={m.cosine:+.3f}")
print("\n:899 (all-caps, attn o_proj) top boosted:")
for m in boosts_c[:5]:
    print(f"  {m.token_str:>12s}  cos={m.cosine:+.3f}")

# %% md-sweep
# ## 3. U-vector addition sweep
#
# For each prefactor α, set:  U[:1127] = U_original[:1127] + α * U[attn.o_proj:899]
# Then visualize model predictions on :1127's firing examples.
#
# We use ComponentTrainer for forward passes so the edit routes through components.
# The trainer snapshots `W_target - V^T @ U` at init (with original U), then each
# forward pass reconstructs the layer output as `V^T @ U_edited + weight_delta`.
# Modifying U.data after init changes only the component contribution.

# %% sweep
u_original_1127 = u_narrative.clone()
eval_examples = firings_narrative[:20]

# Build context-window tensors for baseline caching
# We need stable tensor objects so baselines dict keys (id()) match across calls
eval_tensors: list[Int[Tensor, " seq"]] = []
for ex, fire_pos in eval_examples:
    token_ids, _, _ = extract_context_window(ex, fire_pos)
    eval_tensors.append(torch.tensor(token_ids, device="cuda"))

# Create trainer with original U — snapshot captures unedited weight delta
trainer = ComponentTrainer(em.model, targets={NARRATIVE_KEY: "write"}, lr=1e-3)

# Cache baselines (original U, no edit yet)
baselines: dict[int, Float[Tensor, "seq vocab"]] = {}
with torch.no_grad():
    for tokens_t in eval_tensors:
        baselines[id(tokens_t)] = trainer(tokens_t.unsqueeze(0))[0].softmax(-1)

# Patch build_diffs to use our pre-built tensors instead of re-creating them
def build_diffs_stable(
    trainer: ComponentTrainer,
    baselines: dict[int, Float[Tensor, "seq vocab"]],
    examples: list[tuple[ActivationExample, int]],
    eval_tensors: list[Int[Tensor, " seq"]],
    comp_key: str,
) -> list[ExampleDiff]:
    """Like build_diffs but uses pre-allocated tensors for stable id() keys."""
    module, cidx = parse_component_key(comp_key)
    diffs = []
    with torch.no_grad():
        for (ex, fire_pos), tokens_t in zip(examples, eval_tensors):
            _, firings, _ = extract_context_window(ex, fire_pos)
            token_ids = tokens_t.tolist()

            probs_edit = trainer(tokens_t.unsqueeze(0))[0].softmax(-1)
            probs_base = baselines[id(tokens_t)]
            kl = kl_per_token(probs_edit, probs_base)

            ci_vals = em.get_ci(tokens_t)[module][:, cidx].cpu()
            act_vals = em.get_component_activations(tokens_t, comp_key).cpu()
            spans = tok.get_spans(token_ids)
            fire_set = {i for i, f in enumerate(firings) if f}

            token_diffs = []
            for t in range(len(spans)):
                diff = probs_edit[t] - probs_base[t]
                inc_idx = diff.topk(5).indices
                dec_idx = (-diff).topk(5).indices
                before_idx = probs_base[t].topk(8).indices
                after_idx = probs_edit[t].topk(8).indices
                token_diffs.append(TokenDiff(
                    span=spans[t],
                    kl=kl[t].item(),
                    ci=ci_vals[t].item(),
                    activation=act_vals[t].item(),
                    fires=t in fire_set,
                    topk_before=[
                        (tok.get_tok_display(int(j)), probs_base[t, j].item()) for j in before_idx
                    ],
                    topk_after=[
                        (tok.get_tok_display(int(j)), probs_edit[t, j].item()) for j in after_idx
                    ],
                    top_increases=[
                        (tok.get_tok_display(int(j)), probs_base[t, j].item(), probs_edit[t, j].item())
                        for j in inc_idx
                    ],
                    top_decreases=[
                        (tok.get_tok_display(int(j)), probs_base[t, j].item(), probs_edit[t, j].item())
                        for j in dec_idx
                    ],
                ))
            diffs.append(ExampleDiff(tokens=token_diffs, max_kl=kl.max().item()))
    diffs.sort(key=lambda d: -d.max_kl)
    return diffs


for alpha in PREFACTORS:
    print(f"\n{'='*60}")
    print(f"  α = {alpha}")
    print(f"{'='*60}")

    # Apply the edit: U[down_proj:1127] = U_original + α * U[o_proj:899]
    em.model.components[NARRATIVE_MODULE].U.data[NARRATIVE_IDX] = u_original_1127 + alpha * u_caps

    diffs = build_diffs_stable(trainer, baselines, eval_examples, eval_tensors, NARRATIVE_KEY)

    display(HTML(render_edit_comparison(
        diffs[:15],
        title=f"α = {alpha}: U(down_proj:1127) + {alpha} × U(o_proj:899)",
        subtitle=(
            f"Narrative(down_proj:1127) + {alpha} × AllCaps(o_proj:899). "
            f"Orange = firing position. Hover for logit predictions."
        ),
    )))

# Restore original and clean up
em.model.components[NARRATIVE_MODULE].U.data[NARRATIVE_IDX] = u_original_1127
trainer.cleanup()
print("\nU-vector restored to original.")
