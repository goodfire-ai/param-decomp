"""
Merge iteration with logging support.

This wraps the pure merge_iteration_pure() function and adds WandB/plotting callbacks.
"""

import warnings
from typing import Protocol

import torch
from jaxtyping import Bool, Float
from torch import Tensor
from tqdm import tqdm

from spd.clustering.compute_costs import (
    compute_mdl_cost,
    compute_merge_costs,
    recompute_coacts_merge_pair,
    recompute_coacts_merge_pair_memberships,
)
from spd.clustering.consts import (
    ActivationsTensor,
    BoolActivationsTensor,
    ClusterCoactivationShaped,
    ComponentLabels,
    MergePair,
)
from spd.clustering.math.merge_matrix import GroupMerge
from spd.clustering.merge_config import MergeConfig
from spd.clustering.merge_history import MergeHistory
from spd.clustering.sample_membership import BitsetMembership, compute_coactivation_matrix


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


def merge_iteration(
    merge_config: MergeConfig,
    activations: ActivationsTensor,
    component_labels: ComponentLabels,
    log_callback: LogCallback | None = None,
) -> MergeHistory:
    """
    Merge iteration with optional logging/plotting callbacks.

    This wraps the pure computation with logging capabilities while maintaining
    the same core algorithm logic.
    """

    # compute coactivations
    # --------------------------------------------------
    activation_mask_orig: BoolActivationsTensor | ActivationsTensor | None = (
        activations > merge_config.activation_threshold
        if merge_config.activation_threshold is not None
        else activations
    )
    coact: Float[Tensor, "c c"] = activation_mask_orig.float().T @ activation_mask_orig.float()

    # check shapes
    c_components: int = coact.shape[0]
    assert coact.shape[1] == c_components, "Coactivation matrix must be square"

    # determine number of iterations based on config and number of components
    num_iters: int = merge_config.get_num_iters(c_components)

    # initialize vars
    # --------------------------------------------------
    # start with an identity merge
    current_merge: GroupMerge = GroupMerge.identity(n_components=c_components)

    # initialize variables for the merge process
    k_groups: int = c_components
    current_coact: ClusterCoactivationShaped = coact.clone()
    current_act_mask: Bool[Tensor, "samples k_groups"] = activation_mask_orig.clone()

    # variables we keep track of
    merge_history: MergeHistory = MergeHistory.from_config(
        merge_config=merge_config,
        labels=component_labels,
    )

    # merge iteration
    # ==================================================
    pbar: tqdm[int] = tqdm(
        range(num_iters),
        unit="iter",
        total=num_iters,
    )
    for iter_idx in pbar:
        # compute costs, figure out what to merge
        # --------------------------------------------------
        # HACK: this is messy
        costs: ClusterCoactivationShaped = compute_merge_costs(
            coact=current_coact / current_act_mask.shape[0],
            merges=current_merge,
            alpha=merge_config.alpha,
        )

        merge_pair: MergePair = merge_config.merge_pair_sample(costs)

        # merge the pair
        # --------------------------------------------------
        # we do this *before* logging, so we can see how the sampled pair cost compares
        # to the costs of all the other possible pairs
        current_merge, current_coact, current_act_mask = recompute_coacts_merge_pair(
            coact=current_coact,
            merges=current_merge,
            merge_pair=merge_pair,
            activation_mask=current_act_mask,
        )

        # metrics and logging
        # --------------------------------------------------
        # Store in history
        merge_history.add_iteration(
            idx=iter_idx,
            selected_pair=merge_pair,
            current_merge=current_merge,
        )

        # Compute metrics for logging
        # the MDL loss computed here is the *cost of the current merge*, a single scalar value
        # rather than the *delta in cost from merging a specific pair* (which is what `costs` matrix contains)
        diag_acts: Float[Tensor, " k_groups"] = torch.diag(current_coact)
        mdl_loss: float = compute_mdl_cost(
            acts=diag_acts,
            merges=current_merge,
            alpha=merge_config.alpha,
        )
        mdl_loss_norm: float = mdl_loss / current_act_mask.shape[0]
        # this is the cost for the selected pair
        merge_pair_cost: float = float(costs[merge_pair].item())

        # Update progress bar
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

        # iterate and sanity checks
        # --------------------------------------------------
        k_groups -= 1
        assert current_coact.shape[0] == k_groups, (
            "Coactivation matrix shape should match number of groups"
        )
        assert current_coact.shape[1] == k_groups, (
            "Coactivation matrix shape should match number of groups"
        )
        assert current_act_mask.shape[1] == k_groups, (
            "Activation mask shape should match number of groups"
        )

        # early stopping failsafe
        # --------------------------------------------------
        if k_groups <= 3:
            warnings.warn(
                f"Stopping early at iteration {iter_idx} as only {k_groups} groups left",
                stacklevel=2,
            )
            break

    # finish up
    # ==================================================
    return merge_history


def merge_iteration_memberships(
    merge_config: MergeConfig,
    memberships: list[BitsetMembership],
    n_samples: int,
    component_labels: ComponentLabels,
    log_callback: LogCallback | None = None,
) -> MergeHistory:
    """Exact merge iteration using compressed sample memberships."""
    current_coact: ClusterCoactivationShaped = compute_coactivation_matrix(memberships)

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
