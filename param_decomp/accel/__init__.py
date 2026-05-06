"""Optional acceleration backends for parameter-decomposition scoring."""

from param_decomp.accel.scoring import (
    AccelBackendUnavailable,
    score_rank_one_linear_components,
)

__all__ = ["AccelBackendUnavailable", "score_rank_one_linear_components"]
