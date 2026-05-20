"""The `RunConfig` object: serializable spec for a recipe-mediated PD run.

Pure data. Holds a reloadable ``recipe`` reference plus the three config tiers
(``pd``, ``logging``, ``runtime``). The recipe's own config is dynamically
validated from ``recipe.path`` but stored inside one common top-level
``RunConfig`` shape.

Written to ``run_config.yaml`` beside the checkpoint, passed to the worker, and
re-read on reload. Notebook / script callers do **not** need a ``RunConfig``;
they can call ``optimize`` directly with a ``PDTarget`` + dataloaders.
"""

from pathlib import Path
from typing import Any, Self, override

import yaml
from pydantic import Field, SerializeAsAny, model_validator

from param_decomp.base_config import BaseConfig
from param_decomp.configs import LoggingConfig, PDConfig, RuntimeConfig
from param_decomp.recipes import RunRecipe, load_recipe
from param_decomp.utils.run_utils import generate_run_id

RUN_CONFIG_FILENAME = "run_config.yaml"


class RecipeRef(BaseConfig):
    """Serializable recipe import path plus its typed config."""

    path: str = Field(..., description="Import path of a RunRecipe (`module:attr`).")
    config: SerializeAsAny[BaseConfig]

    @model_validator(mode="before")
    @classmethod
    def validate_recipe_config(cls, data: Any) -> Any:
        if isinstance(data, RecipeRef):
            return data
        assert isinstance(data, dict), f"recipe must be a mapping, got {type(data).__name__}"
        assert data.get("path"), "recipe requires a non-empty `path`"
        recipe = load_recipe(data["path"])
        raw_config = data.get("config", {})
        if isinstance(raw_config, recipe.config_type):
            config = raw_config
        elif isinstance(raw_config, BaseConfig):
            config = recipe.config_type.model_validate(raw_config.model_dump(mode="json"))
        else:
            config = recipe.config_type.model_validate(raw_config)
        return {**data, "config": config}

    def load(self) -> RunRecipe[Any]:
        """Resolve ``path`` to a recipe object."""
        return load_recipe(self.path)


class RunConfig(BaseConfig):
    """Top-level recipe-mediated run config.

    ``run_id`` identifies the output directory and W&B run. Fresh ``RunConfig``
    objects generate one automatically; YAML / dict inputs that already
    contain a value preserve it.

    ``recipe`` is the durable reload hook used to build the target model and
    dataloaders.
    """

    name: str | None = None
    run_id: str = Field(default_factory=lambda: generate_run_id("param_decomp"))
    recipe: RecipeRef
    pd: PDConfig
    logging: LoggingConfig
    runtime: RuntimeConfig
    view_meta: dict[str, Any] = Field(
        default_factory=dict,
        description="Free-form labels for downstream grouping/coloring/reports (e.g. "
        "`{'lr_ratio': 0.1, 'size': 'medium'}`). Populated by sweep generators; surfaced "
        "to W&B under a `view_meta/` prefix.",
    )

    @model_validator(mode="after")
    def validate_metric_overlap(self) -> Self:
        overlap = sorted(set(self.pd.loss_metrics) & set(self.logging.eval_metrics))
        assert not overlap, (
            f"The same metric was set under both pd.loss_metrics and logging.eval_metrics: "
            f"{overlap}. Loss metrics are automatically evaluated; remove the "
            "logging.eval_metrics entry, or move it out of pd.loss_metrics if you want eval-only."
        )
        return self

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunConfig":
        """Parse a dict (e.g. from YAML) into a `RunConfig`.

        The recipe block determines the typed config model used to validate
        ``recipe.config``.
        """
        return cls.model_validate(data)

    @classmethod
    @override
    def from_file(cls, path: Path | str) -> "RunConfig":
        path = Path(path)
        assert path.exists(), f"{RUN_CONFIG_FILENAME} not found at {path}"
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f))

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
