"""Select W&B names without changing the measurements or their local representation."""

from collections.abc import Mapping
from typing import Literal

from pydantic import BaseModel, ConfigDict

from param_decomp.metric_taxonomy import grouped_axis, grouped_metric_key

type MetricSchema = Literal["legacy", "grouped"]


class _RecordedWandbSettings(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    # Runs created before schema selection logged legacy keys.
    metric_schema: MetricSchema = "legacy"


def recorded_metric_schema(config: Mapping[str, object]) -> MetricSchema:
    return _RecordedWandbSettings.model_validate(config.get("wandb", {})).metric_schema


def validate_resume_schema(config: Mapping[str, object], requested: MetricSchema) -> None:
    recorded = recorded_metric_schema(config)
    if recorded != requested:
        raise ValueError(
            f"Cannot resume W&B run with metric_schema={requested!r}; "
            f"it was logged with {recorded!r}. Start a new run or create a saved view."
        )


class MetricNames:
    def __init__(self, schema: MetricSchema):
        self.schema: MetricSchema = schema
        self._sources: dict[str, str] = {}

    def _claim(self, source: str, destination: str) -> str:
        previous = self._sources.get(destination)
        if previous is not None and previous != source:
            raise ValueError(
                f"Metric name collision: {previous!r} and {source!r} map to {destination!r}"
            )
        self._sources[destination] = source
        return destination

    def metric(self, key: str) -> str:
        match self.schema:
            case "legacy":
                return key
            case "grouped":
                return self._claim(key, grouped_metric_key(key).path)

    def axis(self, key: str) -> str:
        match self.schema:
            case "legacy":
                return key
            case "grouped":
                return self._claim(key, grouped_axis(key))

    def record[T](self, record: Mapping[str, T]) -> dict[str, T]:
        return {self.metric(key): value for key, value in record.items()}
