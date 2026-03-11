import tempfile
from pathlib import Path
from typing import Any

import pytest

from spd.clustering.clustering_run_config import ClusteringRunConfig, LoggingIntervals
from spd.clustering.merge_config import MergeConfig
from spd.clustering.scripts.run_clustering import main


@pytest.mark.slow
def test_run_clustering_happy_path(monkeypatch: Any):
    """Test that run_clustering.py runs without errors."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Override SPD_OUT_DIR to use temp directory (avoid creating permanent artifacts)
        temp_path = Path(temp_dir)
        monkeypatch.setattr("spd.settings.SPD_OUT_DIR", temp_path)
        monkeypatch.setattr("spd.utils.run_utils.SPD_OUT_DIR", temp_path)

        config = ClusteringRunConfig(
            model_path="wandb:goodfire/spd/runs/s-a9ad193d",  # A resid_mlp2 run
            batch_size=4,
            dataset_seed=0,
            ensemble_id=None,
            merge_config=MergeConfig(
                activation_threshold=0.01,
                alpha=1.0,
                iters=3,
                merge_pair_sampling_method="range",
                merge_pair_sampling_kwargs={"threshold": 0.05},
            ),
            wandb_project=None,
            wandb_entity="goodfire",
            logging_intervals=LoggingIntervals(
                stat=1,
                tensor=100,
                plot=100,
                artifact=100,
            ),
        )
        main(config)
