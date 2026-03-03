"""Test script to inspect shapes of stochastic masks in the CI optimization code.

Question: Does the stochastic mask cover ALL components (including dead ones with CI=0),
or only the alive components?

We simulate the exact code path from optim_cis.py lines 463-473 without needing a full model.
"""

import torch

from spd.app.backend.optim_cis import AliveComponentInfo, OptimizableCIParams, compute_alive_info
from spd.models.components import make_mask_infos
from spd.routing import AllLayersRouter
from spd.utils.component_utils import calc_stochastic_component_mask_info


# Simulate realistic CI outputs: shape [1, seq, C]
# In a real model, most components at most positions have CI > 0, but some are dead.
seq_len = 5
C = 16  # number of components per layer
layers = ["h.0.mlp.c_fc", "h.0.mlp.down_proj"]

# Create fake CI lower_leaky values: some alive, some dead
torch.manual_seed(42)
ci_lower_leaky: dict[str, torch.Tensor] = {}
ci_pre_sigmoid: dict[str, torch.Tensor] = {}
for layer in layers:
    # Random CI, but zero out ~40% of components to simulate dead ones
    raw = torch.randn(1, seq_len, C)
    pre_sigmoid = raw.clone()
    lower_leaky = torch.sigmoid(raw)
    # Kill ~40% of components
    dead_mask = torch.rand(1, seq_len, C) < 0.4
    lower_leaky[dead_mask] = 0.0
    pre_sigmoid[dead_mask] = 0.0  # dead components have pre_sigmoid=0
    ci_lower_leaky[layer] = lower_leaky
    ci_pre_sigmoid[layer] = pre_sigmoid

print("=== Initial CI shapes and alive/dead counts ===")
for layer, ci in ci_lower_leaky.items():
    n_alive = (ci > 0).sum().item()
    n_dead = (ci == 0).sum().item()
    print(f"  {layer}: shape={ci.shape}, alive={n_alive}, dead={n_dead}")

# Step 1: compute_alive_info (same as optim_cis.py line 418)
alive_info = compute_alive_info(ci_lower_leaky)

# Step 2: create_optimizable_ci_params (same as optim_cis.py line 419)
# We need a mock model for lower_leaky_fn and upper_leaky_fn
# But actually OptimizableCIParams.create_ci_outputs needs a real model.
# Instead, let's just directly call calc_stochastic_component_mask_info with our CI,
# which is exactly what happens at line 468.

print("\n=== Stochastic mask info (line 468 path) ===")
stochastic_mask_infos = calc_stochastic_component_mask_info(
    causal_importances=ci_lower_leaky,
    component_mask_sampling="continuous",
    weight_deltas=None,
    router=AllLayersRouter(),
)

for layer in layers:
    ci = ci_lower_leaky[layer]
    mask_info = stochastic_mask_infos[layer]
    mask = mask_info.component_mask

    dead = ci == 0
    alive = ci > 0
    n_dead = dead.sum().item()
    n_alive = alive.sum().item()

    print(f"\n  {layer}:")
    print(f"    CI shape:   {ci.shape}")
    print(f"    Mask shape: {mask.shape}")
    print(f"    Shapes match: {ci.shape == mask.shape}")

    if n_dead > 0:
        dead_mask_vals = mask[dead]
        print(f"    Dead components ({n_dead}):")
        print(f"      mask values (first 10): {dead_mask_vals[:10].tolist()}")
        print(f"      min={dead_mask_vals.min():.4f}, max={dead_mask_vals.max():.4f}, "
              f"mean={dead_mask_vals.mean():.4f}")
        any_nonzero = (dead_mask_vals > 0).any().item()
        print(f"      Any nonzero? {any_nonzero}")

    if n_alive > 0:
        alive_mask_vals = mask[alive]
        alive_ci_vals = ci[alive]
        print(f"    Alive components ({n_alive}):")
        print(f"      CI range: [{alive_ci_vals.min():.4f}, {alive_ci_vals.max():.4f}]")
        print(f"      mask range: [{alive_mask_vals.min():.4f}, {alive_mask_vals.max():.4f}]")

print("\n=== CI-only mask (line 475 path) ===")
ci_mask_infos = make_mask_infos(component_masks=ci_lower_leaky)
for layer in layers:
    ci = ci_lower_leaky[layer]
    mask = ci_mask_infos[layer].component_mask
    dead = ci == 0
    if dead.sum().item() > 0:
        dead_vals = mask[dead]
        print(f"  {layer}: dead component mask values are all zero? {(dead_vals == 0).all().item()}")

# Now verify: the formula is `ci + (1 - ci) * stochastic_source`
# For dead (ci=0): mask = 0 + 1 * source = source (random [0,1])
# For alive: mask = ci + (1-ci) * source >= ci (always >= ci)
print("\n=== CONCLUSION ===")
print("1. recon_mask_infos has the FULL shape [batch, seq, C] — same as ci_lower_leaky")
print("2. Dead components (CI=0) get RANDOM NONZERO mask values: mask = 0 + 1*rand = rand")
print("3. This means stochastic masking DOES activate components not in the base graph")
print("4. The CI-only mask (mask_type='ci') does NOT — dead components stay at 0")
