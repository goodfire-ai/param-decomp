"""Postprocess pipeline configuration.

PostprocessConfig composes sub-configs for harvest, autointerp, and intruder eval.
Set any section to null to skip that pipeline stage.
"""

from param_decomp.core.base_config import BaseConfig
from param_decomp_goodfire.submit.autointerp import AutointerpSlurmConfig
from param_decomp_goodfire.submit.harvest import HarvestSlurmConfig
from param_decomp_goodfire.submit.intruder import IntruderSlurmConfig


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
    import pathlib

    schema_path = pathlib.Path(__file__).parent / "postprocess.schema.json"
    schema_path.write_text(json.dumps(PostprocessConfig.model_json_schema(), indent=2) + "\n")
