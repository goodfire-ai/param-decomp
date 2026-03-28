"""
Merge iteration with logging support.

This wraps the pure merge_iteration_pure() function and adds WandB/plotting callbacks.
"""

import time
import warnings
from typing import Protocol

import torch
from jaxtyping import Float
from torch import Tensor
from tqdm import tqdm

from spd.clustering.compute_costs import (
    compute_mdl_cost,
    compute_merge_costs,
    recompute_coacts_merge_pair_memberships,
)
from spd.clustering.consts import (
    ClusterCoactivationShaped,
    ComponentLabels,
    MergePair,
)
from spd.clustering.math.merge_matrix import GroupMerge
from spd.clustering.merge_config import MergeConfig
from spd.clustering.merge_history import MergeHistory
from spd.clustering.sample_membership import (
    CompressedMembership,
    compute_coactivation_matrix_from_csr,
    memberships_to_sample_component_csr,
)
from spd.log import logger


def _choose_coact_device(coact: ClusterCoactivationShaped) -> torch.device:
    """Prefer GPU for dense cost-matrix math when enough memory is available."""
    if not torch.cuda.is_available():
        return coact.device

    if coact.device.type == "cuda":
        return coact.device

    free_bytes, _ = torch.cuda.mem_get_info()
    coact_bytes = coact.numel() * coact.element_size()
    # Current coact, a temporary clone during recompute, and the costs tensor dominate.
    required_bytes = coact_bytes * 3 + 512 * 1024**2
    if free_bytes >= required_bytes:
        return torch.device("cuda")

    return coact.device


class LogCallback(Protocol):
    def __call__(
        self,
        current_coact: ClusterCoactivationShaped,
        component_labels: ComponentLabels,
        current_merge: GroupMerge,
        costs: ClusterCoactivationShaped,
        merge_history: MergeHistory,
        iter_idx: int,
        k_groups: int,
        merge_pair_cost: float,
        mdl_loss: float,
        mdl_loss_norm: float,
        diag_acts: Float[Tensor, " k_groups"],
    ) -> None: ...


def merge_iteration_memberships(
    merge_config: MergeConfig,
    memberships: list[CompressedMembership],
    n_samples: int,
    component_labels: ComponentLabels,
    log_callback: LogCallback | None = None,
) -> MergeHistory:
    """Exact merge iteration using compressed sample memberships."""
    csr_start = time.perf_counter()
    component_activity_csr = memberships_to_sample_component_csr(memberships)
    logger.info(
        "Built component activity CSR in "
        f"{time.perf_counter() - csr_start:.2f}s "
        f"(shape={component_activity_csr.shape}, nnz={component_activity_csr.nnz})"
    )

    coact_start = time.perf_counter()
    logger.info(
        "Building coactivation matrix from compressed memberships "
        f"(n_groups={len(memberships)}, n_samples={n_samples})"
    )
    current_coact: ClusterCoactivationShaped = compute_coactivation_matrix_from_csr(
        component_activity_csr
    )
    logger.info(
        "Built coactivation matrix in "
        f"{time.perf_counter() - coact_start:.2f}s "
        f"(shape={tuple(current_coact.shape)})"
    )
    coact_device = _choose_coact_device(current_coact)
    if coact_device != current_coact.device:
        transfer_start = time.perf_counter()
        current_coact = current_coact.to(device=coact_device)
        logger.info(
            "Moved compressed coactivation matrix to "
            f"{coact_device} in {time.perf_counter() - transfer_start:.2f}s"
        )
    else:
        logger.info(f"Keeping compressed coactivation matrix on {current_coact.device}")

    c_components: int = current_coact.shape[0]
    assert current_coact.shape[1] == c_components, "Coactivation matrix must be square"

    num_iters: int = merge_config.get_num_iters(c_components)
    current_merge: GroupMerge = GroupMerge.identity(n_components=c_components)
    current_memberships = memberships.copy()
    k_groups: int = c_components

    merge_history: MergeHistory = MergeHistory.from_config(
        merge_config=merge_config,
        labels=component_labels,
    )

    pbar: tqdm[int] = tqdm(
        range(num_iters),
        unit="iter",
        total=num_iters,
    )
    merge_start = time.perf_counter()
    log_every = min(10, num_iters)
    for iter_idx in pbar:
        costs: ClusterCoactivationShaped = compute_merge_costs(
            coact=current_coact / n_samples,
            merges=current_merge,
            alpha=merge_config.alpha,
        )

        merge_pair: MergePair = merge_config.merge_pair_sample(costs)

        current_merge, current_coact, current_memberships = recompute_coacts_merge_pair_memberships(
            coact=current_coact,
            merges=current_merge,
            merge_pair=merge_pair,
            memberships=current_memberships,
            component_activity_csr=component_activity_csr,
        )

        merge_history.add_iteration(
            idx=iter_idx,
            selected_pair=merge_pair,
            current_merge=current_merge,
        )

        diag_acts: Float[Tensor, " k_groups"] = torch.diag(current_coact)
        mdl_loss: float = compute_mdl_cost(
            acts=diag_acts,
            merges=current_merge,
            alpha=merge_config.alpha,
        )
        mdl_loss_norm: float = mdl_loss / n_samples
        merge_pair_cost: float = float(costs[merge_pair].item())

        pbar.set_description(f"k={k_groups}, mdl={mdl_loss_norm:.4f}, pair={merge_pair_cost:.4f}")

        if log_callback is not None:
            log_callback(
                iter_idx=iter_idx,
                current_coact=current_coact,
                component_labels=component_labels,
                current_merge=current_merge,
                costs=costs,
                merge_history=merge_history,
                k_groups=k_groups,
                merge_pair_cost=merge_pair_cost,
                mdl_loss=mdl_loss,
                mdl_loss_norm=mdl_loss_norm,
                diag_acts=diag_acts,
            )

        if (iter_idx + 1) % log_every == 0 or iter_idx == 0 or iter_idx + 1 == num_iters:
            elapsed = time.perf_counter() - merge_start
            logger.info(
                "Compressed merge progress: "
                f"iter={iter_idx + 1}/{num_iters}, "
                f"elapsed={elapsed:.2f}s, "
                f"sec_per_iter={elapsed / (iter_idx + 1):.4f}, "
                f"k_groups={k_groups - 1}"
            )

        k_groups -= 1
        assert current_coact.shape[0] == k_groups, (
            "Coactivation matrix shape should match number of groups"
        )
        assert current_coact.shape[1] == k_groups, (
            "Coactivation matrix shape should match number of groups"
        )
        assert len(current_memberships) == k_groups, (
            "Membership count should match number of groups"
        )

        if k_groups <= 3:
            warnings.warn(
                f"Stopping early at iteration {iter_idx} as only {k_groups} groups left",
                stacklevel=2,
            )
            break

    return merge_history
