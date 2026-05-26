"""Distributed checkpoint assembly for 3-pool training.

Rank 0 only holds its own block's V/U (and a random-init CI fn). To produce a
checkpoint that matches the schema ``load_component_model_from_checkpoint``
expects (one flat ``state_dict`` with every site's V/U + the trained CI fn),
this module gathers state onto rank 0 from:

  * Every LW block leader → its owned sites' V/U.
  * The CI pool leader → all CI fn params.

Rank 0 assembles the gathered tensors into a temporary "full" ``ComponentModel``
(with all sites in its decomposition targets) and returns that model's
``state_dict()``. The temp model shares ``target_model`` with rank 0's existing
component model — ``ComponentModel.__init__`` doesn't mutate ``target_model``,
so this is safe.

Non-leader ranks no-op. All ranks must enter ``gather_full_state_dict_to_rank0``
in sync so the P2P sends/recvs match up.
"""

# pyright: reportArgumentType=false

import torch
import torch.distributed as dist
import torch.nn as nn
from torch import Tensor

from param_decomp.batch_and_loss_fns import RunBatch
from param_decomp.ci_fns import CiConfig
from param_decomp.ci_sigmoids import SigmoidType
from param_decomp.component_model import ComponentModel
from param_decomp.decomposition_targets import DecompositionTarget
from param_decomp.three_pool.layout import ThreePoolLayout


def gather_full_state_dict_to_rank0(
    layout: ThreePoolLayout,
    component_model: ComponentModel,
    target_model: nn.Module,
    run_batch: RunBatch,
    ci_config: CiConfig,
    sigmoid_type: SigmoidType,
    c_per_site: dict[str, int],
    device: torch.device,
) -> dict[str, Tensor] | None:
    """Collect V/U + CI fn onto rank 0 and return the assembled state_dict.

    Returns the state_dict on rank 0; ``None`` everywhere else. All ranks must
    call this function in sync (the gather uses ordered P2P sends/recvs).
    """
    match layout.my_pool:
        case "layerwise" if layout.my_rank == 0:
            return _rank0_assemble(
                layout=layout,
                local_component_model=component_model,
                target_model=target_model,
                run_batch=run_batch,
                ci_config=ci_config,
                sigmoid_type=sigmoid_type,
                c_per_site=c_per_site,
                device=device,
            )
        case "layerwise" if layout.my_is_block_leader:
            for s in layout.my_owned_sites:
                comp = component_model.components[s]
                dist.send(comp.V.data.contiguous(), dst=0)
                dist.send(comp.U.data.contiguous(), dst=0)
            return None
        case "ci" if layout.my_is_pool_leader:
            assert component_model.ci_fn is not None, "CI pool must keep its CI fn"
            for _, p in component_model.ci_fn.named_parameters():
                dist.send(p.data.contiguous(), dst=0)
            return None
        case _:
            # Non-leader LW ranks, non-leader CI ranks, all PPGD ranks: no-op.
            return None


def _rank0_assemble(
    layout: ThreePoolLayout,
    local_component_model: ComponentModel,
    target_model: nn.Module,
    run_batch: RunBatch,
    ci_config: CiConfig,
    sigmoid_type: SigmoidType,
    c_per_site: dict[str, int],
    device: torch.device,
) -> dict[str, Tensor]:
    """Build a full ComponentModel as an assembly buffer; populate from local
    rank's V/U + recvs; return its state_dict."""
    full_targets = [
        DecompositionTarget(module_path=s, C=c_per_site[s]) for s in layout.world.all_sites
    ]
    # Share target_model with the existing local component_model — ComponentModel
    # __init__ doesn't mutate target_model, so two ComponentModels can coexist
    # over the same target.
    full_cm = ComponentModel(
        target_model=target_model,
        run_batch=run_batch,
        decomposition_targets=full_targets,
        ci_config=ci_config,
        sigmoid_type=sigmoid_type,
    ).to(device)

    # Copy rank 0's own trained V/U into the assembly buffer.
    with torch.no_grad():
        for s in layout.my_owned_sites:
            full_cm.components[s].V.data.copy_(local_component_model.components[s].V.data)
            full_cm.components[s].U.data.copy_(local_component_model.components[s].U.data)

    # Recv V/U from every other LW block leader. Order on both sides must match
    # the iteration order of `bg.owned_sites` for each non-rank-0 block leader.
    for bg in layout.world.layerwise_block_groups:
        if bg.leader == 0:
            continue
        for s in bg.owned_sites:
            V_template = full_cm.components[s].V.data
            U_template = full_cm.components[s].U.data
            V_buf = torch.empty_like(V_template)
            U_buf = torch.empty_like(U_template)
            dist.recv(V_buf, src=bg.leader)
            dist.recv(U_buf, src=bg.leader)
            with torch.no_grad():
                V_template.copy_(V_buf)
                U_template.copy_(U_buf)

    # Recv CI fn params from CI pool leader. Same `named_parameters()` iteration
    # order on both sides since the CI fn is constructed from the same config.
    ci_leader = layout.world.ci_ranks[0]
    assert full_cm.ci_fn is not None, "checkpoint reconstruction needs a CI fn"
    for _, p in full_cm.ci_fn.named_parameters():
        buf = torch.empty_like(p.data)
        dist.recv(buf, src=ci_leader)
        with torch.no_grad():
            p.data.copy_(buf)

    state_dict = {k: v.cpu() for k, v in full_cm.state_dict().items()}
    del full_cm
    torch.cuda.empty_cache()
    return state_dict
