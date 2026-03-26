"""Write-vector editing for SPD components.

Each SPD component is a rank-1 adapter: V[:, c] @ U[c, :]. The write vector (U row)
determines what the component contributes to the residual stream.

u_replaced: context manager that swaps a U row and runs through the component path.

Usage:
    model, tok, config = load_model("wandb:goodfire/spd/s-55ea3f9b")

    unembed = model.target_model.lm_head.weight[token_id].detach()
    new_u = -3.0 * unembed / unembed.norm()
    with u_replaced(model, "h.2.mlp.down_proj", 2359, new_u) as forward_fn:
        logits = forward_fn(tokens.unsqueeze(0))
"""

from contextlib import contextmanager

import torch
from jaxtyping import Float
from torch import Tensor

from spd.models.component_model import ComponentModel
from spd.models.components import ComponentsMaskInfo, make_mask_infos
from spd.utils.general_utils import get_obj_device


def all_ones_mask_infos(model: ComponentModel) -> dict[str, ComponentsMaskInfo]:
    device = get_obj_device(model)
    component_masks = {}
    weight_deltas_and_masks = {}
    for module_name in model.target_module_paths:
        C = model.module_to_c[module_name]
        component_masks[module_name] = torch.ones((C,), device=device)
        wd = model.calc_weight_deltas()[module_name]
        wdm = torch.ones((C,), device=device)
        weight_deltas_and_masks[module_name] = (wd, wdm)
    return make_mask_infos(component_masks, weight_deltas_and_masks)


@contextmanager
def u_replaced(
    model: ComponentModel,
    module_name: str,
    u_idx: int,
    new_u: Float[Tensor, " d_out"],
):
    """Replace U[u_idx] with new_u. Forward runs through the component path (all-ones masks)."""
    comp = model.components[module_name]
    old_u = comp.U.data[u_idx].clone()
    assert old_u.shape == new_u.shape

    comp.U.data[u_idx] = new_u
    mask_infos = all_ones_mask_infos(model)

    def forward_fn(tokens: Tensor) -> Tensor:
        return model(tokens, mask_infos=mask_infos)

    try:
        yield forward_fn
    finally:
        comp.U.data[u_idx] = old_u
