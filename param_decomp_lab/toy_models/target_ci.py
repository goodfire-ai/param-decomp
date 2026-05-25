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
    """Permute columns to make the matrix as close to identity as possible (greedy).

    Args:
        ci_vals: Causal importance values matrix.

    Returns:
        Tuple of the permuted matrix and the permutation indices.
    """
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
    """Permute columns to make the matrix as close to identity as possible (Hungarian).

    Args:
        ci_vals: Causal importance values matrix.

    Returns:
        Tuple of the permuted matrix and the permutation indices.
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
    """Permute columns to make the matrix as close to identity as possible.

    Args:
        ci_vals: Causal importance values matrix.
        method: Algorithm to use for permutation. "hungarian" is optimal but O(n^3);
            "greedy" is faster but suboptimal; "auto" picks Hungarian for matrices
            with min dimension < 500 and greedy otherwise.

    Returns:
        Tuple of the permuted matrix and the permutation indices.
    """
    if method == "hungarian" or (method == "auto" and min(ci_vals.shape) < 500):
        return permute_to_identity_hungarian(ci_vals)
    else:
        return permute_to_identity_greedy(ci_vals)


def permute_to_dense(
    ci_vals: Float[Tensor, "batch C"],
) -> tuple[Float[Tensor, "batch C"], Int[Tensor, " C"]]:
    """Permute columns by total mass, densest first.

    Args:
        ci_vals: Causal importance values matrix.

    Returns:
        Tuple of the permuted matrix (densest columns first) and the permutation indices.
    """
    if ci_vals.ndim != 2:
        raise ValueError(f"Matrix must have 2 dimensions, got {ci_vals.ndim}")

    # Sort columns by total mass in descending order
    column_sums = ci_vals.sum(dim=0)
    perm_indices = torch.argsort(column_sums, descending=True)

    return ci_vals[:, perm_indices], perm_indices


class TargetCIPattern(ABC):
    """Base class for target sparsity patterns."""

    def _verify_inputs(self, ci_array: Float[Tensor, "batch C"]) -> None:
        """Verify that the input is a 2D tensor.

        Args:
            ci_array: Causal importance matrix to validate.
        """
        if ci_array.ndim != 2:
            raise ValueError(f"Expected 2D tensor, got shape {ci_array.shape}")

    @abstractmethod
    def distance_from(self, ci_array: Float[Tensor, "batch C"], tolerance: float = 0.1) -> int:
        """Count elements that deviate from the expected pattern beyond tolerance.

        The tolerance threshold avoids sensitivity to small values from inactive
        components: elements are counted as off only if they deviate from the
        expected value by more than ``tolerance``.

        Args:
            ci_array: Causal importance matrix to evaluate.
            tolerance: Per-element deviation threshold.

        Returns:
            Number of elements that violate the pattern.
        """
        pass


class IdentityCIPattern(TargetCIPattern):
    """Expect a one-to-one feature-to-component mapping.

    Each feature should activate exactly one component (up to permutation).
    Counts elements that violate this pattern beyond the tolerance threshold.

    Attributes:
        n_features: Expected number of features (rows).
        apply_permutation: Whether to align columns to the identity before scoring.
        method: Permutation algorithm: ``"hungarian"``, ``"greedy"``, or ``"auto"``.
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
    """Expect exactly ``k`` active components (columns with strong activations).

    Error is computed against the column-sorted matrix: the first ``k`` columns
    contribute one error per missing strong activation (below ``min_entries``),
    and the remaining columns contribute one error per weak activation above
    tolerance (they should be fully inactive).

    Attributes:
        k: Number of columns that should be active.
        min_entries: Minimum number of strong activations (> 1 - tolerance)
            required for a column to be considered active.
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
    """Collection of expected patterns for different modules in a model.

    Keys of ``module_targets`` may be exact module names or fnmatch-style
    patterns (e.g. ``"layers.0.mlp_in"``, ``"layers.*.mlp_in"``,
    ``"layers.*.mlp_*"``). Patterns are expanded at runtime against actual
    module names; the first matching pattern wins for each module.

    Attributes:
        module_targets: Mapping from module name pattern to target pattern.
    """

    def __init__(self, module_targets: dict[str, TargetCIPattern]):
        """Initialize the solution with pattern mappings.

        Args:
            module_targets: Mapping from module name pattern (exact or fnmatch-style,
                e.g. ``"layers.*.mlp_in"``) to its target pattern.
        """
        self.module_targets = module_targets

    def expand_module_targets(self, module_names: list[str]) -> dict[str, TargetCIPattern]:
        """Resolve patterns to a concrete module-name to pattern mapping.

        Args:
            module_names: Concrete module names to match against the stored patterns.

        Returns:
            Mapping from each matched module name to its target pattern.
        """
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
        """Sum the per-module pattern distances across all modules.

        Args:
            ci_arrays: Mapping from module name to causal importance matrix.
            tolerance: Per-element deviation threshold.

        Returns:
            Total number of off elements across all modules.
        """
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
    """Compute total and per-module target-solution distance metrics.

    Args:
        causal_importances: Mapping from module name to causal importance tensor.
        target_solution: Target solution to compare against.
        tolerance: Per-element deviation threshold for pattern matching.

    Returns:
        Mapping with ``"total"`` and ``"total_0p2"`` aggregate distances plus one
        entry per matched module name.
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
    """Build a ``TargetCISolution`` from config-style pattern specifications.

    Args:
        identity_ci: Identity-pattern specs, each with ``layer_pattern`` and ``n_features``.
        dense_ci: Dense-pattern specs, each with ``layer_pattern`` and ``k``.

    Returns:
        Assembled ``TargetCISolution``, or ``None`` when both lists are empty.
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
