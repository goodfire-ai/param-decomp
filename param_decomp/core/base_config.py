"""`BaseConfig` (pydantic `BaseModel` with `extra="forbid"`, `frozen=True`, YAML/JSON
round-trip) and `Probability` (annotated `float` in `[0, 1]`).
"""

import json
from functools import cached_property
from pathlib import Path
from typing import Annotated, ClassVar, Self

import yaml
from annotated_types import Ge, Le
from pydantic import BaseModel, ConfigDict

Probability = Annotated[float, Ge(0), Le(1)]
"""A float constrained to `[0, 1]` for pydantic validation."""


class BaseConfig(BaseModel):
    """Pydantic `BaseModel` tailored for configs.

    `extra="forbid"`, `frozen=True`, plus `from_file` / `to_file` JSON/YAML round-trip
    helpers.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid", frozen=True, ignored_types=(cached_property,)
    )

    @classmethod
    def from_file(cls, path: Path | str) -> Self:
        """Load and validate a config from a `.json`, `.yaml`, or `.yml` file.

        Validation errors are re-raised with a note that includes the source path and the
        parsed data for debugging.
        """
        if isinstance(path, str):
            path = Path(path)

        match path:
            case Path() if path.suffix == ".json":
                data = json.loads(path.read_text())
            case Path() if path.suffix in [".yaml", ".yml"]:
                data = yaml.safe_load(path.read_text())
            case _:
                raise ValueError(f"Only (.json, .yaml, .yml) files are supported, got {path}")

        try:
            cfg = cls.model_validate(data)
        except Exception as e:
            e.add_note(f"Error validating config {cls=} from path `{path.as_posix()}`\n{data = }")
            raise e
        return cfg

    def to_file(self, path: Path | str) -> None:
        """Serialize this config to `path`.

        `.json` writes indent-2 JSON; `.yaml` / `.yml` writes a JSON-mode YAML dump.
        Creates parent directories as needed.
        """
        if isinstance(path, str):
            path = Path(path)

        path.parent.mkdir(parents=True, exist_ok=True)
        match path.suffix:
            case ".json":
                path.write_text(self.model_dump_json(indent=2))
            case ".yaml" | ".yml":
                path.write_text(yaml.dump(self.model_dump(mode="json")))
            case _:
                raise ValueError(f"Only (.json, .yaml, .yml) files are supported, got {path}")
