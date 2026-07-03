"""Merge iteration with optional logging callback."""

import time
from typing import Protocol

import numpy as np
from jaxtyping import Float
from tqdm import tqdm

from param_decomp.log import logger
from param_decomp_lab.clustering.compute_costs import (
    compute_mdl_cost,
    compute_merge_costs,
    recompute_coacts_merge_pair_memberships,
)
from param_decomp_lab.clustering.math.merge_matrix import GroupMerge
from param_decomp_lab.clustering.merge_config import MergeConfig
from param_decomp_lab.clustering.merge_history import MergeHistory
from param_decomp_lab.clustering.sample_membership import (
    CompressedMembership,
    compute_coactivation_matrix_from_csr,
    memberships_to_sample_component_csr,
)
from param_decomp_lab.clustering.types import (
    ClusterCoactivationShaped,
    ComponentLabels,
    MergePair,
)


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
        diag_acts: Float[np.ndarray, " k_groups"],
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

    pbar: tqdm[int] = tqdm(range(num_iters), unit="iter", total=num_iters)
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

        diag_acts: Float[np.ndarray, " k_groups"] = np.diag(current_coact)
        mdl_loss: float = compute_mdl_cost(
            acts=diag_acts,
            merges=current_merge,
            alpha=merge_config.alpha,
        )
        mdl_loss_norm: float = mdl_loss / n_samples
        merge_pair_cost: float = float(costs[merge_pair])

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
            logger.info(f"Stopping early at iteration {iter_idx} as only {k_groups} groups left")
            break

    return merge_history
