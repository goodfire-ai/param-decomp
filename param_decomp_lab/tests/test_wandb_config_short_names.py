from param_decomp_config.wandb_config import METRIC_SHORT_NAMES
from param_decomp_lab.eval_metrics import metric_short_names


def test_config_short_name_table_matches_metric_registry():
    """The torch-free `METRIC_SHORT_NAMES` (consumed by the JAX trainer for an identical
    wandb config key layout) must stay in lockstep with the torch metric classes."""
    assert metric_short_names() == METRIC_SHORT_NAMES
