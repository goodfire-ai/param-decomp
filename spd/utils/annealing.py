"""Utilities for annealing scalar values over the course of training."""


def linearly_anneal_value(
    current_frac_of_training: float,
    initial_value: float,
    start_frac: float,
    final_value: float | None,
    end_frac: float,
) -> float:
    """Linearly anneal a value between two training fractions.

    Returns ``initial_value`` before ``start_frac`` and ``final_value`` at/after ``end_frac``.
    In between, linearly interpolates. If ``final_value`` is None or ``start_frac >= 1.0``,
    no annealing is applied and ``initial_value`` is returned.
    """
    if final_value is None or start_frac >= 1.0:
        return initial_value

    assert end_frac >= start_frac, f"end_frac ({end_frac}) must be >= start_frac ({start_frac})"

    if current_frac_of_training < start_frac:
        return initial_value
    if current_frac_of_training >= end_frac:
        return final_value
    progress = (current_frac_of_training - start_frac) / (end_frac - start_frac)
    return initial_value + (final_value - initial_value) * progress
