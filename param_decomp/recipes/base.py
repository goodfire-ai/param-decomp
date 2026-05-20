"""Reloadable run recipes.

A recipe is the durable extension point for recipe-mediated PD runs. It owns the
experiment-specific config needed to reconstruct runtime objects after launch:
target model, train loader, and eval loader. Core run configs stay one common
shape; the recipe config is validated dynamically from ``recipe.path``.
"""

from importlib import import_module
from typing import Any, ClassVar, Protocol, cast

from torch.utils.data import DataLoader

from param_decomp.base_config import BaseConfig
from param_decomp.models.batch_and_loss_fns import PDTarget
from param_decomp.utils.distributed_utils import DistributedState


class RunRecipe[ConfigT: BaseConfig](Protocol):
    """Converts a typed recipe config into runtime PD objects."""

    name: ClassVar[str]

    @property
    def config_type(self) -> type[ConfigT]:
        """Pydantic model type used to validate ``recipe.config``."""
        ...

    def build_target(self, cfg: ConfigT) -> PDTarget:
        """Build the target model bundle from upstream."""
        ...

    def build_train_loader(
        self,
        cfg: ConfigT,
        *,
        device: str,
        batch_size: int,
        seed: int,
        dist_state: DistributedState | None = None,
    ) -> DataLoader[Any]:
        """Build the train dataloader for the saved recipe."""
        ...

    def build_eval_loader(
        self,
        cfg: ConfigT,
        *,
        device: str,
        batch_size: int,
        seed: int,
        dist_state: DistributedState | None = None,
    ) -> DataLoader[Any]:
        """Build the eval dataloader for the saved recipe."""
        ...


def load_recipe(recipe_path: str) -> RunRecipe[Any]:
    """Load a recipe object or no-arg recipe class from ``"module:attr"``."""
    module_path, sep, attr = recipe_path.partition(":")
    if sep == "":
        raise ValueError(f"Recipe path must be of the form 'module:attr', got {recipe_path!r}")
    recipe = getattr(import_module(module_path), attr)
    if isinstance(recipe, type):
        recipe = recipe()
    return cast(RunRecipe[Any], recipe)
