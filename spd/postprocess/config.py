"""Postprocess pipeline configuration.

PostprocessConfig composes sub-configs for harvest, attributions, autointerp,
and intruder eval. Set any section to null to skip that pipeline stage.
"""

from typing import Any, override

from spd.autointerp.config import AutointerpSlurmConfig
from spd.base_config import BaseConfig
from spd.dataset_attributions.config import AttributionsSlurmConfig
from spd.graph_interp.config import GraphInterpSlurmConfig
from spd.harvest.config import HarvestSlurmConfig, IntruderSlurmConfig, SPDHarvestConfig


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
        is_not_spd = not isinstance(self.harvest.config.method_config, SPDHarvestConfig)
        if expects_attributions and is_not_spd:
            raise ValueError("Attributions only work for SPD decompositions")
        if self.graph_interp is not None and self.attributions is None:
            raise ValueError("Graph interp requires attributions")


if __name__ == "__main__":
    import json

    with open("spd/postprocess/postprocess.schema.json", "w") as f:
        json.dump(PostprocessConfig.model_json_schema(), f, indent=2)
