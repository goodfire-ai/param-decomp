import tempfile
from pathlib import Path
from typing import Any

import pytest

from spd.clustering.clustering_run_config import ClusteringRunConfig
from spd.clustering.harvest_config import HarvestConfig
from spd.clustering.merge_config import MergeConfig
from spd.clustering.scripts.run_clustering import main


@pytest.mark.slow
def test_run_clustering_happy_path(monkeypatch: Any):
    """Test that run_clustering.py runs without errors."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        monkeypatch.setattr("spd.settings.SPD_OUT_DIR", temp_path)
        monkeypatch.setattr("spd.utils.run_utils.SPD_OUT_DIR", temp_path)

        config = ClusteringRunConfig(
            harvest=HarvestConfig(
                model_path="wandb:goodfire/spd/runs/s-a9ad193d",
                batch_size=4,
                activation_threshold=0.01,
            ),
            merge=MergeConfig(
                alpha=1.0,
                iters=3,
                merge_pair_sampling_method="range",
                merge_pair_sampling_kwargs={"threshold": 0.05},
            ),
            wandb_project=None,
        )
        main(config, run_id="c-test")
