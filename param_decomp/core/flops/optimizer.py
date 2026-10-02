"""Useful optimizer arithmetic over logical parameters, independent of placement.

Contractions use conventional two-FLOP multiply-adds. Scalar addition, multiplication,
division and square root each count as one operation. Comparisons, casts, random draws,
indexing, communication and scalar schedules are excluded. Scalar coefficients can be
shared and folded; these are algebraic counts, not an execution or bitwise-parity model.
"""

from collections.abc import Callable
from dataclasses import dataclass
from math import isfinite
from typing import Literal

from param_decomp.core.ci_fn.architecture import CIFnArchitectureFootprint
from param_decomp.core.components import SiteSpec, component_parameters
from param_decomp.core.configs import (
    AdamPGDConfig,
    AdamWOptimizerConfig,
    AnyOptimizerConfig,
    BatchSourceShape,
    CIMaskedReconLossConfig,
    CIMaskedReconSubsetLossConfig,
    FaithfulnessLossConfig,
    ImportanceMinimalityLossConfig,
    MergedStochasticSubsetPooledPPGDReconLossConfig,
    MergedStochasticSubsetPPGDReconLossConfig,
    MomentumSgdPGDConfig,
    MuonOptimizerConfig,
    NonlinearityLocalityLossConfig,
    PDConfig,
    PersistentPGDReconLossConfig,
    PGDReconLossConfig,
    PGDReconSubsetLossConfig,
    SgdPGDConfig,
    StochasticReconLossConfig,
    StochasticReconSubsetLossConfig,
    TargetedPDConfig,
    UnmaskedReconLossConfig,
)
from param_decomp.core.flops.types import MatrixParameters, ParameterCensus
from param_decomp.core.model import PositionAxis, Positioned, Positionless
from param_decomp.core.schedule import get_scheduled_value

type ArithmeticDtype = Literal["float32", "bfloat16"]


@dataclass(frozen=True)
class OptimizerFlopsTerm:
    name: str
    contraction_flops: float
    elementwise_flops: float
    dtype: ArithmeticDtype
    n_repetitions: int

    def __post_init__(self) -> None:
        if (
            not self.name
            or not all(
                isfinite(value) and value >= 0
                for value in (self.contraction_flops, self.elementwise_flops)
            )
            or self.n_repetitions < 1
        ):
            raise ValueError(
                "Optimizer terms require a name, nonnegative costs and a positive count"
            )

    @property
    def total(self) -> float:
        return self.n_repetitions * (self.contraction_flops + self.elementwise_flops)


@dataclass(frozen=True)
class OptimizerFlops:
    terms: tuple[OptimizerFlopsTerm, ...]

    @property
    def contraction_flops(self) -> float:
        return sum(term.n_repetitions * term.contraction_flops for term in self.terms)

    @property
    def elementwise_flops(self) -> float:
        return sum(term.n_repetitions * term.elementwise_flops for term in self.terms)

    @property
    def total(self) -> float:
        return self.contraction_flops + self.elementwise_flops


def newton_schulz_flops(
    matrix: MatrixParameters, n_steps: int, dtype: ArithmeticDtype, name: str
) -> OptimizerFlopsTerm:
    """Evaluate X' = (aI + bA + cA²)X, A = XXᵀ, with the smaller Gram matrix.

    A and A² are symmetric: each needs only its triangle. Their final product with X
    needs every entry. This conventional contraction model exploits symmetry, without
    asserting a lower bound over every possible matrix multiplication algorithm.
    """
    if n_steps < 1:
        raise ValueError("Newton-Schulz requires at least one iteration")
    n_rows, n_columns = sorted((matrix.n_rows, matrix.n_columns))
    n_triangle_elements = n_rows * (n_rows + 1) // 2
    contractions = (
        2 * n_triangle_elements * n_columns
        + 2 * n_triangle_elements * n_rows
        + 2 * n_rows * n_rows * n_columns
    )
    polynomial = 3 * n_triangle_elements + n_rows
    # The Frobenius norm uses N squares, N-1 additions, sqrt, epsilon and N divisions.
    normalization = 3 * n_rows * n_columns + 1
    return OptimizerFlopsTerm(
        name,
        matrix.n_matrices * n_steps * contractions,
        matrix.n_matrices * (normalization + n_steps * polynomial),
        dtype,
        1,
    )


def parameter_optimizer_flops(
    name: str, optimizer: AnyOptimizerConfig, parameters: ParameterCensus
) -> tuple[OptimizerFlopsTerm, ...]:
    """Count updates with common scalar coefficients folded before elementwise work.

    Adam costs 3 operations for its first moment, 4 for its second, and 5 for the
    normalized update and parameter addition. Nesterov adds 3; nonzero decay adds 1.
    Muon costs 3 for momentum, 3 for Nesterov, and 2 for applying the scaled update.
    """
    decay = int(optimizer.weight_decay != 0)
    terms: list[OptimizerFlopsTerm] = []
    match optimizer:
        case AdamWOptimizerConfig():
            arithmetic = parameters.n_parameters * (12 + decay)
        case MuonOptimizerConfig():
            arithmetic = (
                sum(matrix.n_parameters for matrix in parameters.matrices) * (8 + decay)
                + 15 * parameters.n_vector_parameters
            )
            terms.extend(
                newton_schulz_flops(
                    matrix, optimizer.ns_steps, optimizer.ns_dtype, f"{name}/newton_schulz/{index}"
                )
                for index, matrix in enumerate(parameters.matrices)
            )
    terms.insert(0, OptimizerFlopsTerm(f"{name}/update", 0, arithmetic, "float32", 1))
    if optimizer.grad_clip_norm is not None and parameters.n_parameters:
        # N squares, N-1 additions, sqrt, epsilon, division, and N rescalings.
        terms.append(
            OptimizerFlopsTerm(
                f"{name}/gradient_clip", 0, 3 * parameters.n_parameters + 2, "float32", 1
            )
        )
    return tuple(terms)


def _n_source_copies(shape: BatchSourceShape, batch_size: int, positions: PositionAxis) -> int:
    match positions:
        case Positioned(n_positions=length):
            if length < 1:
                raise ValueError("Source FLOPs require positive sequence length")
        case Positionless():
            length = 1
            if shape == "bsc":
                raise ValueError("Position-indexed sources require a positioned target")
    match shape:
        case "bc":
            return batch_size
        case "bsc":
            return batch_size * length


def prepare_optimizer_flops(
    pd: PDConfig | TargetedPDConfig,
    sites: tuple[SiteSpec, ...],
    ci_fn_arch: CIFnArchitectureFootprint,
    positions: PositionAxis,
) -> Callable[[int], OptimizerFlops]:
    """Resolve fixed optimizer work before evaluating scheduled adversarial batch draws."""
    component_census = component_parameters(sites)
    terms = list(parameter_optimizer_flops("components", pd.components_optimizer, component_census))
    terms.extend(
        parameter_optimizer_flops("ci", pd.ci_fn_optimizer, ci_fn_arch.parameter_census(sites))
    )
    match pd:
        case TargetedPDConfig():
            if pd.ci_scaled_weight_decay is not None:
                scale_arithmetic = 2 * sum(site.C for site in sites)
                terms.append(
                    OptimizerFlopsTerm(
                        "components/ci_scaled_weight_decay",
                        0,
                        component_census.n_parameters + scale_arithmetic,
                        "float32",
                        1,
                    )
                )
        case PDConfig():
            pass
    source_width = sum(site.C + 1 for site in sites)
    adversaries: list[Callable[[int], tuple[OptimizerFlopsTerm, ...]]] = []
    for loss in pd.loss_metrics:
        name = f"adversary/{loss.name or loss.type}"
        match loss:
            case PGDReconLossConfig() | PGDReconSubsetLossConfig():
                if loss.n_steps:
                    n_parameters = source_width * _n_source_copies(
                        loss.source_shape, pd.batch_size, positions
                    )
                    terms.append(
                        OptimizerFlopsTerm(name, 0, 2 * n_parameters, "float32", loss.n_steps)
                    )
            case (
                PersistentPGDReconLossConfig()
                | MergedStochasticSubsetPPGDReconLossConfig()
                | MergedStochasticSubsetPooledPPGDReconLossConfig()
            ):
                adversaries.append(
                    _prepare_persistent_optimizer_flops(
                        loss, source_width, pd.batch_size, positions, pd.steps
                    )
                )
            case (
                FaithfulnessLossConfig()
                | ImportanceMinimalityLossConfig()
                | NonlinearityLocalityLossConfig()
                | CIMaskedReconLossConfig()
                | CIMaskedReconSubsetLossConfig()
                | StochasticReconLossConfig()
                | StochasticReconSubsetLossConfig()
                | UnmaskedReconLossConfig()
            ):
                pass
    fixed_terms = tuple(terms)

    def at_step(step: int) -> OptimizerFlops:
        return OptimizerFlops(
            fixed_terms + tuple(term for adversary in adversaries for term in adversary(step))
        )

    return at_step


def _prepare_persistent_optimizer_flops(
    loss: PersistentPGDReconLossConfig
    | MergedStochasticSubsetPPGDReconLossConfig
    | MergedStochasticSubsetPooledPPGDReconLossConfig,
    source_width: int,
    batch_size: int,
    positions: PositionAxis,
    n_steps: int,
) -> Callable[[int], tuple[OptimizerFlopsTerm, ...]]:
    """Unsampled particles have zero gradients but their existing moments still evolve.

    Adam uses seven moment operations and five update operations; zero gradients
    reduce its moment work to two multiplications. Momentum similarly loses its
    gradient addition. SGD leaves unsampled particles unchanged. Moment/source storage
    determines their arithmetic precision; normalized updates use the fp32 LR.
    """
    match loss:
        case MergedStochasticSubsetPooledPPGDReconLossConfig():
            n_warmup_parameters = batch_size * source_width
            n_parameters = n_warmup_parameters * loss.pool.size_per_batch_element
        case PersistentPGDReconLossConfig() | MergedStochasticSubsetPPGDReconLossConfig():
            n_parameters = source_width * _n_source_copies(loss.source_shape, batch_size, positions)
            n_warmup_parameters = n_parameters
    n_warmup_updates = loss.n_warmup_steps * n_warmup_parameters
    n_total_updates = (loss.n_warmup_steps + 1) * n_parameters
    match loss.source_dtype:
        case "float32" | "uint16":
            storage_arithmetic_dtype: ArithmeticDtype = "float32"
        case "bfloat16":
            storage_arithmetic_dtype = "bfloat16"
    match loss.optimizer:
        case AdamPGDConfig():
            active_storage, inactive_storage = 8, 3
            active_fp32, inactive_fp32 = 4, 4
        case SgdPGDConfig():
            active_storage, inactive_storage = 1, 0
            active_fp32, inactive_fp32 = 1, 0
        case MomentumSgdPGDConfig():
            active_storage, inactive_storage = 1, 1
            active_fp32, inactive_fp32 = 3, 2
    name = f"adversary/{loss.name or loss.type}"

    def terms_for_probability(probability: float) -> tuple[OptimizerFlopsTerm, ...]:
        n_active_updates = n_warmup_updates + probability * n_warmup_parameters
        n_inactive_updates = n_total_updates - n_active_updates
        return (
            OptimizerFlopsTerm(
                f"{name}/source_arithmetic",
                0,
                active_storage * n_active_updates + inactive_storage * n_inactive_updates,
                storage_arithmetic_dtype,
                1,
            ),
            OptimizerFlopsTerm(
                f"{name}/update_arithmetic",
                0,
                active_fp32 * n_active_updates + inactive_fp32 * n_inactive_updates,
                "float32",
                1,
            ),
        )

    match loss:
        case PersistentPGDReconLossConfig():
            fixed_terms = terms_for_probability(1.0)
            return lambda step: fixed_terms
        case (
            MergedStochasticSubsetPPGDReconLossConfig()
            | MergedStochasticSubsetPooledPPGDReconLossConfig()
        ):
            fraction = loss.adv_fraction

            def at_step(step: int) -> tuple[OptimizerFlopsTerm, ...]:
                probability = get_scheduled_value(step, n_steps, fraction)
                return terms_for_probability(probability)

            return at_step
