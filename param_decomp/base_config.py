import json
from pathlib import Path
from typing import ClassVar, Self

import yaml
from pydantic import BaseModel, ConfigDict


class BaseConfig(BaseModel):
    """Pydantic BaseModel suited for configs.

    Uses the pydantic `model_config` to enforce `extra="forbid"` and `frozen=True` and add loading
    and saving from/to YAML, JSON.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    # TODO: add a "config_type" field, which is set to the class name, so that when loading a config we can check whether the config type matches the expected class

    @classmethod
    def from_file(cls, path: Path | str) -> Self:
        """Load config from path to a JSON or YAML file."""
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
        """Save config to file (format inferred from extension)."""
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
