"""Run-file naming shared by the JAX trainer and its consumers.

The `ExperimentConfig` YAML schema lives in `param_decomp_lab.experiments.config`. The JAX
trainer stamps its config under this filename (the `PDAdapter` reload contract).
"""

EXPERIMENT_CONFIG_FILENAME = "experiment_config.yaml"
