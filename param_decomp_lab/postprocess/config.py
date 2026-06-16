"""Postprocess pipeline configuration.

PostprocessConfig composes sub-configs for harvest, autointerp, and intruder eval.
Set any section to null to skip that pipeline stage.
"""

from param_decomp_config.base import BaseConfig
from param_decomp_lab.autointerp.config import AutointerpSlurmConfig
from param_decomp_lab.harvest.config import HarvestSlurmConfig, IntruderSlurmConfig


class PostprocessConfig(BaseConfig):
    """Top-level config for the unified postprocessing pipeline.

    Composes sub-configs for each pipeline stage. Only `harvest` is required;
    omit a downstream stage (or set it to null) to skip it.

    Dependency graph:
        harvest                 (GPU array -> merge, PD-only)
        ├── intruder eval       (CPU, label-free, depends on harvest merge)
        └── autointerp          (CPU, LLM calls, depends on harvest merge)
            ├── detection
            └── fuzzing
    """

    harvest: HarvestSlurmConfig
    autointerp: AutointerpSlurmConfig | None = None
    intruder: IntruderSlurmConfig | None = None


if __name__ == "__main__":
    import json

    with open("param_decomp_lab/postprocess/postprocess.schema.json", "w") as f:
        json.dump(PostprocessConfig.model_json_schema(), f, indent=2)
