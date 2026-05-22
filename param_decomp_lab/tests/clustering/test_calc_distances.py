from param_decomp_lab.clustering.merge_config import MergeConfig
from param_decomp_lab.clustering.merge_history import MergeHistory, MergeHistoryEnsemble
from param_decomp_lab.clustering.types import ComponentLabels


def test_merge_history_normalization_happy_path():
    """Test that the normalization part of calc_distances.py works without errors"""

    # Create test merge histories
    config = MergeConfig(
        iters=3,
        alpha=1.0,
    )

    histories = []
    for _idx in range(2):
        history = MergeHistory.from_config(
            merge_config=config,
            labels=ComponentLabels([f"comp{j}" for j in range(4)]),
        )
        histories.append(history)

    # Test ensemble creation
    ensemble = MergeHistoryEnsemble(data=histories)
    assert len(ensemble.data) == 2

    # Test normalization
    normalized_array, metadata = ensemble.normalized()
    assert normalized_array is not None
    assert metadata is not None
