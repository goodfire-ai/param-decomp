# spd.editing

Component-level model editing for VPD decompositions.

## Setup

```python
from spd.editing import EditableModel, generate, measure_kl, measure_token_probs
from spd.harvest.repo import HarvestRepo
from spd.autointerp.repo import InterpRepo

em, tok = EditableModel.from_wandb("wandb:goodfire/spd/s-892f140b")
harvest = HarvestRepo("s-892f140b")
interp = InterpRepo("s-892f140b")
```

## Finding components

By autointerp label:
```python
from spd.editing import search_interpretations
matches = search_interpretations(harvest, interp, r"male pronoun")
# -> [ComponentMatch(key='h.1.attn.v_proj:52', label='male pronouns', ...)]
```

By output token PMI (best for ablation targets):
```python
from spd.editing import search_by_token_pmi
he_id = tok.encode("he")
matches = search_by_token_pmi(harvest, he_id, side="output", min_pmi=1.0)
```

By circuit optimization across examples:
```python
examples = [(tokens1, target_pos1), (tokens2, target_pos2), ...]
components = em.find_components_by_examples(examples, optim_steps=100)
# -> [('h.1.attn.v_proj:52', 0.9), ('h.1.mlp.down_proj:798', 0.8), ...]
```

## Inspecting components

```python
from spd.editing import inspect_component
data = inspect_component(harvest, interp, "h.1.mlp.down_proj:798", tok)
# Prints: label, input/output PMI tokens, activation examples
```

Component geometry:
```python
vecs = em.get_component_vectors("h.1.mlp.down_proj:798")  # read (V) and write (U) vectors
alignment = em.component_alignment("h.1.attn.o_proj:82", "h.1.mlp.c_fc:144")  # cosine, percentile
boosted, suppressed = em.unembed_alignment("h.1.mlp.down_proj:798", tok)  # top logit-lens tokens
```

## Editing (runtime masks)

```python
# 0.0 = ablate, 2.0 = boost
edit_fn = em.make_edit_fn({"h.1.mlp.down_proj:798": 0.0, "h.1.attn.v_proj:52": 0.0})

# Generate with edits
text = generate(edit_fn, tokens, tok)

# Measure effect
effect = measure_kl(em, edit_fn, eval_seqs)
print(f"KL={effect.mean_kl:.3f}, PPL: {effect.baseline_ppl:.1f} -> {effect.edited_ppl:.1f}")

# Token group probability shifts
shifts = measure_token_probs(em, edit_fn, eval_seqs, {
    "he": tok.encode("he"),
    "she": tok.encode("she"),
})
print(f"P(he) change: {shifts['he'].change_pct:+.1f}%")
```

CI-conditional editing (only edit where component is active):
```python
edit_fn = em.make_edit_fn({"h.1.mlp.down_proj:798": 0.0}, ci_threshold=0.1)
```

## Permanent weight editing

```python
clean_em = em.without_components(["h.1.mlp.down_proj:798"])
# Returns a new EditableModel with rank-1 subtraction baked into weights
text = generate(clean_em, tokens, tok)
```

## Circuit analysis

```python
circuit = em.optimize_circuit(tokens, target_position=15, target_token=tok.encode("he")[0])
em.print_circuit(circuit, tokens, tok, interp=interp)
# Prints: edges, node CI, component labels
```
