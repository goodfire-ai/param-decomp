"""Target patterns for evaluating causal importance matrices.

This module provides abstractions for testing whether learned sparsity patterns
match expected target solutions in toy models:

- TargetCIPattern classes define expected sparsity patterns (Identity, DenseColumns)
- TargetCISolution maps model components to their expected patterns
- Evaluation uses a discrete distance metric that counts elements deviating beyond
  a tolerance threshold, making it robust to small values from inactive components
"""

import fnmatch
from abc import ABC, abstractmethod
from typing import Literal, override

import torch
from jaxtyping import Float, Int
from torch import Tensor

from param_decomp_lab.toy_models._linear_sum_assignment import linear_sum_assignment


def permute_to_identity_greedy(
    ci_vals: Float[Tensor, "batch C"],
) -> tuple[Float[Tensor, "batch C"], Int[Tensor, " C"]]:
    """Greedy column permutation toward identity. Returns `(permuted, perm_indices)`."""
    if ci_vals.ndim != 2:
        raise ValueError(f"Mask must have 2 dimensions, got {ci_vals.ndim}")

    batch, C = ci_vals.shape
    effective_rows = min(batch, C)

    perm = []
    used = set()
    for i in range(effective_rows):
        sorted_indices = torch.argsort(ci_vals[i, :], descending=True)
        chosen = next(
            (col.item() for col in sorted_indices if col.item() not in used),
            sorted_indices[0].item(),
        )
        perm.append(chosen)
        used.add(chosen)

    # Add remaining columns
    remaining = sorted(set(range(C)) - used)
    perm.extend(remaining)

    perm_indices = torch.tensor(perm, device=ci_vals.device, dtype=torch.long)
    return ci_vals[:, perm_indices], perm_indices


def permute_to_identity_hungarian(
    ci_vals: Float[Tensor, "batch C"],
) -> tuple[Float[Tensor, "batch C"], Int[Tensor, " C"]]:
    """Optimal column permutation toward identity (Hungarian).

    Returns `(permuted, perm_indices)`.
    """
    if ci_vals.ndim != 2:
        raise ValueError(f"Mask must have 2 dimensions, got {ci_vals.ndim}")

    batch, C = ci_vals.shape
    device = ci_vals.device
    effective_rows = min(batch, C)

    # Hungarian algorithm on the effective_rows x C submatrix
    cost_matrix = -ci_vals[:effective_rows].detach().cpu().numpy()
    _, col_indices = linear_sum_assignment(cost_matrix)

    # Build complete permutation
    assigned_cols = set(col_indices.tolist())
    unassigned_cols = sorted(set(range(C)) - assigned_cols)

    perm_list = list(col_indices) + unassigned_cols
    perm_indices = torch.tensor(perm_list, device=device, dtype=torch.long)

    return ci_vals[:, perm_indices], perm_indices


def permute_to_identity(
    ci_vals: Float[Tensor, "batch C"],
    method: Literal["hungarian", "greedy", "auto"] = "auto",
) -> tuple[Float[Tensor, "batch C"], Int[Tensor, " C"]]:
    """Permute columns toward identity.

    `"hungarian"` is optimal but `O(n^3)`; `"greedy"` is faster but suboptimal; `"auto"`
    picks Hungarian when `min(shape) < 500`.
    """
    if method == "hungarian" or (method == "auto" and min(ci_vals.shape) < 500):
        return permute_to_identity_hungarian(ci_vals)
    else:
        return permute_to_identity_greedy(ci_vals)


def permute_to_dense(
    ci_vals: Float[Tensor, "batch C"],
) -> tuple[Float[Tensor, "batch C"], Int[Tensor, " C"]]:
    """Permute columns by total mass, densest first. Returns `(permuted, perm_indices)`."""
    if ci_vals.ndim != 2:
        raise ValueError(f"Matrix must have 2 dimensions, got {ci_vals.ndim}")

    # Sort columns by total mass in descending order
    column_sums = ci_vals.sum(dim=0)
    perm_indices = torch.argsort(column_sums, descending=True)

    return ci_vals[:, perm_indices], perm_indices


class TargetCIPattern(ABC):
    """Base class for target sparsity patterns."""

    def _verify_inputs(self, ci_array: Float[Tensor, "batch C"]) -> None:
        if ci_array.ndim != 2:
            raise ValueError(f"Expected 2D tensor, got shape {ci_array.shape}")

    @abstractmethod
    def distance_from(self, ci_array: Float[Tensor, "batch C"], tolerance: float = 0.1) -> int:
        """Count elements deviating from the expected pattern by more than `tolerance`.

        The tolerance avoids sensitivity to small values from inactive components.
        """
        pass


class IdentityCIPattern(TargetCIPattern):
    """Expect a one-to-one feature-to-component mapping.

    Each feature should activate exactly one component (up to permutation); counts
    elements violating this pattern beyond `tolerance`.
    """

    def __init__(
        self,
        n_features: int,
        apply_permutation: bool = True,
        method: Literal["hungarian", "greedy", "auto"] = "auto",
    ):
        self.n_features = n_features
        self.apply_permutation = apply_permutation
        self.method = method

    @override
    def _verify_inputs(self, ci_array: Float[Tensor, "batch C"]) -> None:
        super()._verify_inputs(ci_array)
        n, c = ci_array.shape
        if n != self.n_features:
            raise ValueError(f"Expected {self.n_features} features, got {n}")
        if c < self.n_features:
            raise ValueError(f"Expected at least {self.n_features} components, got {c}")

    @override
    def distance_from(self, ci_array: Float[Tensor, "batch C"], tolerance: float = 0.1) -> int:
        self._verify_inputs(ci_array)
        if self.apply_permutation:
            # Hungarian algorithm is O(n^3) complexity. Sample CPU runtimes: ~0.15s for 250x250, ~1.5s for 500x500, ~26s for 1000x1000.
            # By default, we use Hungarian for small matrices (min dimension < 500) and greedy for larger matrices.
            if self.method == "hungarian" or (self.method == "auto" and min(ci_array.shape) < 500):
                ci_array = permute_to_identity_hungarian(ci_array)[0]
            else:
                ci_array = permute_to_identity_greedy(ci_array)[0]

        size = min(ci_array.shape)
        # Off-diagonal errors + on-diagonal errors
        mask = torch.ones_like(ci_array, dtype=torch.bool)
        mask[:size, :size].fill_diagonal_(False)
        off_diag_errors = torch.sum(ci_array[mask] > tolerance)
        on_diag_errors = torch.sum(torch.diag(ci_array[:size, :size]) < (1 - tolerance))
        return int(off_diag_errors + on_diag_errors)


class DenseCIPattern(TargetCIPattern):
    """Expect exactly `k` active columns.

    Error against the column-sorted matrix: the first `k` columns contribute one error
    per missing strong activation (column needs `>= min_entries` entries above
    `1 - tolerance`); the rest contribute one error per weak activation above tolerance
    (should be fully inactive).
    """

    def __init__(self, k: int, min_entries: int = 1):
        self.k = k
        self.min_entries = min_entries

    @override
    def _verify_inputs(self, ci_array: Float[Tensor, "batch C"]) -> None:
        super()._verify_inputs(ci_array)
        _, c = ci_array.shape
        if c < self.k:
            raise ValueError(f"Expected at least {self.k} columns, got {c}")

    @override
    def distance_from(self, ci_array: Float[Tensor, "batch C"], tolerance: float = 0.1) -> int:
        self._verify_inputs(ci_array)
        sorted_ci = permute_to_dense(ci_array)[0]

        strong_activations_per_column = (sorted_ci >= 1 - tolerance).sum(dim=0)
        missing_strong_activations = torch.clamp(
            self.min_entries - strong_activations_per_column, min=0
        )
        first_k_column_error = missing_strong_activations[: self.k].sum().item()

        weak_activations_per_column = (sorted_ci > tolerance).sum(dim=0)
        inactive_column_error = weak_activations_per_column[self.k :].sum().item()

        return int(first_k_column_error + inactive_column_error)


class TargetCISolution:
    """Expected target patterns per module.

    Keys may be exact module names or fnmatch-style patterns (e.g. `"layers.*.mlp_in"`).
    Patterns are expanded against actual module names at runtime; the first matching
    pattern wins per module.
    """

    def __init__(self, module_targets: dict[str, TargetCIPattern]):
        self.module_targets = module_targets

    def expand_module_targets(self, module_names: list[str]) -> dict[str, TargetCIPattern]:
        """Resolve patterns to a concrete `{module_name: target_pattern}` map."""
        result = {}
        for name in module_names:
            for pattern, target in self.module_targets.items():
                if fnmatch.fnmatch(name, pattern):
                    result[name] = target
                    break

        return result

    def distance_from(
        self, ci_arrays: dict[str, Float[Tensor, "batch C"]], tolerance: float = 0.1
    ) -> int:
        """Sum per-module pattern distances across all modules."""
        expanded_targets = self.expand_module_targets(list(ci_arrays.keys()))

        return sum(
            target.distance_from(ci_arrays[name], tolerance)
            for name, target in expanded_targets.items()
        )


def compute_target_metrics(
    causal_importances: dict[str, Float[Tensor, "batch C"]],
    target_solution: TargetCISolution,
    tolerance: float = 0.1,
) -> dict[str, float]:
    """Per-module + aggregate target-solution distances.

    Returns `{"total", "total_0p2", <per_module_name>: ...}` (`total_0p2` is
    `tolerance=0.2`).
    """
    metrics = {}

    # Total error across all modules
    metrics["total"] = target_solution.distance_from(causal_importances, tolerance)
    metrics["total_0p2"] = target_solution.distance_from(causal_importances, 0.2)

    # Per-module errors
    expanded_targets = target_solution.expand_module_targets(list(causal_importances.keys()))
    for module_name, pattern in expanded_targets.items():
        module_error = pattern.distance_from(causal_importances[module_name], tolerance)
        metrics[module_name] = module_error

    return metrics


def make_target_ci_solution(
    identity_ci: list[dict[str, str | int]] | None = None,
    dense_ci: list[dict[str, str | int]] | None = None,
) -> TargetCISolution | None:
    """Build a `TargetCISolution` from config-style specs.

    Each `identity_ci` entry needs `{layer_pattern, n_features}`; each `dense_ci` entry
    needs `{layer_pattern, k}`. Returns `None` when both lists are empty.
    """
    if not identity_ci and not dense_ci:
        return None

    module_targets = {}

    if identity_ci:
        for spec in identity_ci:
            module_targets[spec["layer_pattern"]] = IdentityCIPattern(
                n_features=int(spec["n_features"])
            )

    if dense_ci:
        for spec in dense_ci:
            module_targets[spec["layer_pattern"]] = DenseCIPattern(k=int(spec["k"]))

    return TargetCISolution(module_targets)
