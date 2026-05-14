import tempfile
from pathlib import Path
from typing import Any

import pytest

from param_decomp.clustering.clustering_run_config import ClusteringRunConfig
from param_decomp.clustering.harvest_config import HarvestConfig
from param_decomp.clustering.merge_config import MergeConfig
from param_decomp.clustering.scripts.run_clustering import main


@pytest.mark.slow
@pytest.mark.requires_wandb
def test_run_clustering_happy_path(monkeypatch: Any):
    """Test that run_clustering.py runs without errors."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        monkeypatch.setattr("param_decomp.settings.PARAM_DECOMP_OUT_DIR", temp_path)
        monkeypatch.setattr("param_decomp.utils.run_utils.PARAM_DECOMP_OUT_DIR", temp_path)

        config = ClusteringRunConfig(
            harvest=HarvestConfig(
                model_path="wandb:goodfire/param-decomp/runs/p-2bbbb3b5",
                batch_size=2,
                n_tokens=16,
                n_tokens_per_seq=4,
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
