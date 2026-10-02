"""Evaluation through the CI lifecycle interface."""

from jaxtyping import Array

from param_decomp.core.ci_fn.interface import CI, CIFn
from param_decomp.core.model import ComponentActivations
from param_decomp.sequence import SequenceLayout


def evaluate_ci_from_captures[Conditioning](
    ci_fn: CIFn[Conditioning],
    captures: dict[str, Array],
    conditioning: Conditioning,
    components: ComponentActivations,
    *,
    sequence: SequenceLayout | None,
    remat: bool,
) -> CI:
    """Evaluate from exactly the physical target captures requested by CI."""
    _validate_captures(ci_fn, captures)
    return ci_fn(captures, conditioning, components, sequence=sequence, remat=remat)


def _validate_captures[Conditioning](ci_fn: CIFn[Conditioning], captures: dict[str, Array]) -> None:
    expected = ci_fn.capture_keys
    if captures.keys() != expected:
        raise ValueError(
            f"CI captures must match its request: missing={sorted(expected - captures.keys())}, "
            f"unexpected={sorted(captures.keys() - expected)}"
        )
