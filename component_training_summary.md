# Component Write-Vector Training: Results Summary

**Date**: 2026-03-25
**Branch**: `feature/component-trainer`
**Notebook**: `notebooks/2026-03-25-13-10_component_training_results.ipynb`
**Model**: Jose (`s-55ea3f9b`), `pile_llama_simple_mlp-4L`, tokenizer `EleutherAI/gpt-neox-20b`

## What we did

Each SPD component is a rank-1 adapter: `V[:, c] @ U[c, :]`. The read vector (V column) determines *when* it fires; the write vector (U row) determines *what* it writes to the residual stream. We trained write vectors to redirect component output to a target token, measuring on-target accuracy and off-target collateral.

**Target behavior**: emoticon completion (component `h.2.mlp.down_proj:2359`, "punctuation marks starting an emoticon"). Target token: `'o'` (token id 80). The component fires on colon/semicolon contexts — roughly 50/50 genuine emoticons vs non-emoticon colons (code, URLs, timestamps).

## Key code

```python
from spd.editing import EditableModel
from spd.editing.component_trainer import ComponentTrainer

em, tok = EditableModel.from_wandb("wandb:goodfire/spd/s-55ea3f9b")

# ComponentTrainer freezes everything, unfreezes specific V columns / U rows
# via gradient masks. Forward uses all-ones masks + snapshotted weight delta.
trainer = ComponentTrainer(em.model, targets={"h.2.mlp.down_proj:2359": "write"}, lr=1e-3)

# Train: mutate next-token at firing positions to target, CE loss only there
for step in range(100):
    logits = trainer(tokens.unsqueeze(0))
    loss = F.cross_entropy(logits[0, firing_positions], target_tokens)
    trainer.step(loss)

trainer.cleanup()
```

**Important gotcha**: `ComponentTrainer` snapshots `weight_delta = W_target - V^T @ U` at init. Creating a *new* trainer from an already-trained model absorbs the edit into the snapshot, collapsing the diff to zero. Always cache baselines before training with the *same* trainer instance, and for scale sweeps, keep one trainer alive and modify `U.data` directly.

### Modules

- `spd/editing/component_trainer.py` — `ComponentTrainer` class
- `spd/editing/compare.py` — `train_and_compare()`: train + eval + per-token diff collection
- `spd/editing/viz.py` — `render_edit_comparison()`: interactive HTML heatmap for notebooks

### Evaluation metrics

We measure at three off-target regimes, all excluding firing positions:

| Regime | What it measures |
|--------|-----------------|
| **Post-fire** | KL at positions 1-5 after a firing (excluding other firings in the sequence) |
| **Surrounding** | KL at all non-firing positions within activation-example windows |
| **Global** | KL on random text unrelated to the component |

On-target: P(target_token) and rank(target_token) at firing positions.

## Results

### 1. Write-only training works with very few examples

| n train | P('o') | Surrounding KL | Global KL | surr/glob |
|---------|--------|---------------|-----------|-----------|
| 1       | 73%    | 0.006         | 0.005     | 1.3x      |
| 2       | 82%    | 0.009         | 0.006     | 1.6x      |
| 4       | 83%    | 0.010         | 0.006     | 1.7x      |
| 8       | 85%    | 0.011         | 0.006     | 1.8x      |
| 16      | 92%    | 0.025         | 0.008     | 3.0x      |
| 32      | 97%    | 0.184         | 0.033     | 5.5x      |

Sweet spot is **n=4-8**: 83-85% on-target with essentially zero off-target (surrounding KL barely above global baseline). More examples buy diminishing returns on accuracy but increasing collateral. n=32 blows up off-target damage.

### 2. The edit direction is consistent and interpretable

10 independent single-example runs produce U deltas with:
- **Mean pairwise cosine: 0.63** (min 0.51, max 0.78)
- Each delta has 0.78-0.86 cosine with the mean direction

The training isn't memorizing examples — it's finding a consistent direction in weight space that is a property of the component's geometry.

### 3. The delta is the negated unembed direction

The mean trained delta is **rank 1 most anti-aligned** with `unembed('o')` out of 50,277 tokens:
- cos(delta, unembed('o')) = -0.50 (n=1) to -0.66 (n=16), runner-up at -0.24
- The mechanism is double-negation: component has **negative activations** at firing positions, anti-aligned write vector → positive logit contribution for 'o'

### 4. Analytical replacement: no training needed

Setting `U_row = U_original - 3.0 * unembed('o') / |unembed('o')|` (no training, no data) matches trained n=8:

| Method | P('o') | Surrounding KL | Global KL |
|--------|--------|---------------|-----------|
| Trained n=8 | 85% | 0.011 | 0.006 |
| Analytical | 86% | 0.013 | 0.007 |

The trained delta's orthogonal-to-unembed components provide marginal cleanup (~15% less surrounding KL) but aren't needed for the on-target effect.

### 5. The edit exceeds the emoji probability ceiling

At firing positions, the model's baseline probability mass on emoji-like characters is **51%**. The edit pushes P('o') to 85-95% — it's not just redistributing emoji probability, it's pulling mass from non-emoji tokens. The component's write vector has sufficient influence to override the model's original prediction.

### 6. Component sweep: accuracy/locality tradeoff

Across 7 `h.2.mlp.down_proj` emoticon components:

| Component | P('o') | Post-fire KL | Global KL | Label |
|-----------|--------|-------------|-----------|-------|
| :1672 | **98%** | 0.015 | 0.010 | emoticon continuations after : or = |
| :3327 | 95% | 0.207 | 0.010 | emoticon completions after : or ; |
| :2359 | 94% | 0.098 | **0.007** | punctuation starting an emoticon |
| :3382 | 93% | 0.053 | 0.010 | emoticon continuations after :;= |
| :3290 | 90% | **0.012** | 0.013 | emoticons and smiley faces |

Components show a clear accuracy vs locality tradeoff. :2359 is the best overall (lowest global KL), :3290 has lowest post-fire bleed, :1672 has highest accuracy.

## What to investigate next (for Lucius / other behaviors)

### Choosing a target behavior

Pick a component (or cluster of components) for a different behavior. Good candidates:
- Search autointerp labels: `search_interpretations(harvest, interp, r"your_regex")`
- Look for components in `down_proj` layers (write-side of MLP) — these are the most directly editable
- The component should fire reasonably often (>10 firing positions in harvest data) and have a clear "target token" you want to redirect to

### Workflow

1. **Find components**: search autointerp labels, pick a cluster in `down_proj`
2. **Pick target token**: what should the component predict instead? Get its token id from `tok.get_tok_display(id)` / `tok._tok.encode("your_token")`
3. **Quick test**: `train_and_compare()` with 6 train / 30 held-out, write-only mode
4. **Blast radius sweep**: vary n_examples (1,2,4,8,16,32) × U-scale (0 to 1.5), measure on-fire P(target), surrounding KL, global KL
5. **Analytical test**: try `U = U_orig - α * unembed(target) / |unembed(target)|`, sweep α from 0 to 6
6. **Emoji mass equivalent**: measure baseline probability mass on "related" tokens at firing positions — this is your ceiling for "just redistributing within-category mass"

### Things to watch for

- **Snapshot gotcha**: never create a new `ComponentTrainer` from a model that was already trained — the snapshot absorbs the edit. Keep one trainer alive.
- **Firing position quality**: harvest activation examples include all positions where CI > threshold, which may include noise. Consider LLM-labeling firing contexts (we used haiku, ~$0.01 for 1100 examples in 6 batches of 200).
- **The 1.7x activation ratio**: for :2359, emoticon vs non-emoticon firings only differ by ~1.7x in activation magnitude. This limits how much U-scaling can separate on-target from off-target within the firing regime. Write-only editing affects all firing contexts equally in direction, differing only by activation magnitude.
- **n=32 cliff**: off-target damage spikes dramatically beyond n~16. The learned direction starts drifting (0.84 cosine with n=1 delta vs 0.97 for n=4).
