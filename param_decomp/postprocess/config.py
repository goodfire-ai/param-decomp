"""Postprocess pipeline configuration.

PostprocessConfig composes sub-configs for harvest, attributions, autointerp,
and intruder eval. Set any section to null to skip that pipeline stage.
"""

from typing import Any, override

from param_decomp.autointerp.config import AutointerpSlurmConfig
from param_decomp.base_config import BaseConfig
from param_decomp.dataset_attributions.config import AttributionsSlurmConfig
from param_decomp.graph_interp.config import GraphInterpSlurmConfig
from param_decomp.harvest.config import (
    HarvestSlurmConfig,
    IntruderSlurmConfig,
    ParamDecompHarvestConfig,
)


class PostprocessConfig(BaseConfig):
    """Top-level config for the unified postprocessing pipeline.

    Composes sub-configs for each pipeline stage. Set a section to null
    to skip that stage entirely.

    Dependency graph:
        harvest (GPU array -> merge)
        ├── intruder eval    (CPU, depends on harvest merge, label-free)
        └── autointerp       (depends on harvest merge)
            ├── interpret
            │   ├── detection
            │   └── fuzzing
        attributions (GPU array -> merge, depends on harvest merge)
    """

    harvest: HarvestSlurmConfig
    autointerp: AutointerpSlurmConfig | None
    intruder: IntruderSlurmConfig | None
    attributions: AttributionsSlurmConfig | None
    graph_interp: GraphInterpSlurmConfig | None

    @override
    def model_post_init(self, __context: Any) -> None:
        expects_attributions = self.attributions is not None
        is_not_pd = not isinstance(self.harvest.config.method_config, ParamDecompHarvestConfig)
        if expects_attributions and is_not_pd:
            raise ValueError("Attributions only work for PD decompositions")
        if self.graph_interp is not None and self.attributions is None:
            raise ValueError("Graph interp requires attributions")


if __name__ == "__main__":
    import json

    with open("param_decomp/postprocess/postprocess.schema.json", "w") as f:
        json.dump(PostprocessConfig.model_json_schema(), f, indent=2)
