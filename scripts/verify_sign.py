"""Verify attribution sign for a specific node.

Run goodfire/spd/s-275c8f21, prompt "The Princess lost her crown."
Node 3.mlp.down, component 1538 at seq pos 2 ("lost").

Question: Large negative attribution to output tokens " her", " his", " their".
Ablating component increases probability. Is this consistent?
"""

import torch

from spd.app.backend.app_tokenizer import AppTokenizer
from spd.app.backend.compute import _setup_embed_hook
from spd.models.component_model import ComponentModel, OutputWithCache, SPDRunInfo
from spd.models.components import make_mask_infos
from spd.topology import TransformerTopology
from spd.utils.general_utils import bf16_autocast

WANDB_PATH = "wandb:goodfire/spd/s-275c8f21"
PROMPT = "The Princess lost her crown."
TARGET_LAYER = "h.3.mlp.down_proj"  # concrete module path for 3.mlp.down
TARGET_SEQ = 2  # "lost"
TARGET_COMP = 1538

device = "cuda"

# Load model
run_info = SPDRunInfo.from_path(WANDB_PATH)
model = ComponentModel.from_run_info(run_info).to(device).eval()
topology = TransformerTopology(model.target_model)

# Load tokenizer
assert run_info.config.tokenizer_name is not None
app_tok = AppTokenizer.from_pretrained(run_info.config.tokenizer_name)
token_ids = app_tok.encode(PROMPT)
tokens_display = [app_tok.hf_tokenizer.decode([t]) for t in token_ids]
print(f"Tokens: {tokens_display}")
print(f"Token IDs: {token_ids}")

tokens_tensor = torch.tensor([token_ids], device=device)
n_seq = len(token_ids)

# Get number of components per layer
n_components = {name: comp.C for name, comp in model.components.items()}
print(f"Components per layer: {n_components}")

# 1. Baseline: full model forward (no masking = target model output)
print("\n=== Baseline (no masking) ===")
with torch.no_grad(), bf16_autocast():
    baseline_logits = model(tokens_tensor)
    baseline_probs = torch.softmax(baseline_logits[0].float(), dim=-1)

target_tokens = [" her", " his", " their"]
for tok_str in target_tokens:
    tok_id = app_tok.encode(tok_str)
    assert len(tok_id) == 1, f"Expected single token for '{tok_str}', got {tok_id}"
    tok_id = tok_id[0]
    prob = baseline_probs[TARGET_SEQ, tok_id].item()
    logit = baseline_logits[0, TARGET_SEQ, tok_id].item()
    print(f"  '{tok_str}' (id={tok_id}): logit={logit:.4f}, prob={prob:.6f}")

# 2. Ablation: zero out component 1538 at seq pos 2 of h.3.mlp.down_proj
print(f"\n=== Ablation: zero component {TARGET_COMP} at seq {TARGET_SEQ} of {TARGET_LAYER} ===")
with torch.no_grad(), bf16_autocast():
    # Create masks: all ones except target component at target position
    ablation_masks = {}
    for name, comp in model.components.items():
        mask = torch.ones(1, n_seq, comp.C, device=device)
        if name == TARGET_LAYER:
            mask[0, TARGET_SEQ, TARGET_COMP] = 0.0
        ablation_masks[name] = mask

    mask_infos = make_mask_infos(component_masks=ablation_masks)
    ablated_logits = model(tokens_tensor, mask_infos=mask_infos)
    ablated_probs = torch.softmax(ablated_logits[0].float(), dim=-1)

for tok_str in target_tokens:
    tok_id = app_tok.encode(tok_str)[0]
    base_prob = baseline_probs[TARGET_SEQ, tok_id].item()
    abl_prob = ablated_probs[TARGET_SEQ, tok_id].item()
    base_logit = baseline_logits[0, TARGET_SEQ, tok_id].item()
    abl_logit = ablated_logits[0, TARGET_SEQ, tok_id].item()
    print(f"  '{tok_str}' (id={tok_id}):")
    print(f"    baseline: logit={base_logit:.4f}, prob={base_prob:.6f}")
    print(f"    ablated:  logit={abl_logit:.4f}, prob={abl_prob:.6f}")
    print(f"    delta:    logit={abl_logit - base_logit:+.4f}, prob={abl_prob - base_prob:+.6f}")

# 3. Compute grad*act attribution from 3.mlp.down to output logits
print(f"\n=== Attribution: grad*act from {TARGET_LAYER} comp {TARGET_COMP} to output logits ===")

embed_path = topology.path_schema.embedding_path
unembed_path = topology.path_schema.unembed_path

# Setup forward pass with caching (same as compute.py)
weight_deltas = model.calc_weight_deltas()
weight_deltas_and_masks = {
    k: (v, torch.ones(tokens_tensor.shape, device=device)) for k, v in weight_deltas.items()
}

# All-ones masks (unmasked forward)
unmasked_component_masks = {
    name: torch.ones(1, n_seq, comp.C, device=device) for name, comp in model.components.items()
}
unmasked_masks = make_mask_infos(
    component_masks=unmasked_component_masks,
    weight_deltas_and_masks=weight_deltas_and_masks,
)

embed_hook, embed_cache = _setup_embed_hook()
embed_handle = topology.embedding_module.register_forward_hook(embed_hook, with_kwargs=True)

with torch.enable_grad(), bf16_autocast():
    comp_output: OutputWithCache = model(
        tokens_tensor, mask_infos=unmasked_masks, cache_type="component_acts"
    )

embed_handle.remove()

cache = comp_output.cache
cache[f"{embed_path}_post_detach"] = embed_cache[0]
cache[f"{unembed_path}_pre_detach"] = comp_output.output

# The output logits
logits_for_grad = cache[f"{unembed_path}_pre_detach"]  # [1, seq, vocab]
source_post_detach = cache[f"{TARGET_LAYER}_post_detach"]  # [1, seq, C]

print(f"  logits shape: {logits_for_grad.shape}")
print(f"  source_post_detach shape: {source_post_detach.shape}")
act_val = source_post_detach[0, TARGET_SEQ, TARGET_COMP].item()
print(f"  source activation (post_detach[0, {TARGET_SEQ}, {TARGET_COMP}]) = {act_val:.6f}")

for tok_str in target_tokens:
    tok_id = app_tok.encode(tok_str)[0]

    grad = torch.autograd.grad(
        outputs=logits_for_grad[0, TARGET_SEQ, tok_id],
        inputs=source_post_detach,
        retain_graph=True,
    )[0]

    grad_val = grad[0, TARGET_SEQ, TARGET_COMP].item()
    attribution = grad_val * act_val

    print(f"  '{tok_str}' (id={tok_id}):")
    print(f"    grad = {grad_val:.6f}")
    print(f"    act  = {act_val:.6f}")
    print(f"    attr = grad*act = {attribution:.6f}")

# 4. Comparison: predicted vs actual effect
print("\n=== Comparison: predicted vs actual delta from ablation ===")
print("  (attribution = component's contribution to logit)")
print("  (predicted delta from ablation = -attribution)")
print()
for tok_str in target_tokens:
    tok_id = app_tok.encode(tok_str)[0]

    grad = torch.autograd.grad(
        outputs=logits_for_grad[0, TARGET_SEQ, tok_id],
        inputs=source_post_detach,
        retain_graph=True,
    )[0]

    grad_val = grad[0, TARGET_SEQ, TARGET_COMP].item()
    attribution = grad_val * act_val

    base_logit = baseline_logits[0, TARGET_SEQ, tok_id].item()
    abl_logit = ablated_logits[0, TARGET_SEQ, tok_id].item()
    actual_delta = abl_logit - base_logit

    print(f"  '{tok_str}' (id={tok_id}):")
    print(f"    attribution     = {attribution:+.6f}")
    print(f"    predicted delta = {-attribution:+.6f} (= -attribution)")
    print(f"    actual delta    = {actual_delta:+.4f} (from ablation)")
    sign_match = (attribution > 0) == (actual_delta < 0) or attribution == 0
    print(f"    sign consistent: {'YES' if sign_match else 'NO'}")
