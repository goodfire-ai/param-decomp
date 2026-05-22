"""Pydantic schemas for sweep YAML.

A sweep YAML carries:
  * fixed knobs (model spec + runtime knobs like qos, time, wandb_project)
  * a ``defaults`` block specifying partial-or-full sweep-point fields
  * a ``grid`` of axes whose values override the defaults; the cartesian
    product of the grid defines the materialized sweep points.

Each materialized point is a :class:`SweepPoint` with all required fields set.
"""

import itertools
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PositiveInt, model_validator


class ModelSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(..., description="HF model id (also used as tokenizer id).")
    n_layers: PositiveInt
    vocab_size: PositiveInt


class RuntimeSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    qos: str | None = None
    time: str = "00:25:00"
    wandb_project: str = "param-decomp"


class TopologySpec(BaseModel):
    """Per-point topology: site grouping × within-group DDP factor."""

    model_config = ConfigDict(extra="forbid")
    grouping: Literal["fused", "attn_mlp", "per_site"]
    ddp: PositiveInt = 1
    blocks_per_group: PositiveInt = 1
    pool_b: PositiveInt | None = None
    use_fused_kl: bool = True

    @model_validator(mode="after")
    def _validate_blocks_per_group(self) -> "TopologySpec":
        # blocks_per_group only makes sense for fused — attn_mlp and per_site
        # are already finer-than-block partitions of a single layer.
        if self.blocks_per_group != 1:
            assert self.grouping == "fused", (
                f"blocks_per_group>1 only valid for grouping=fused, got grouping={self.grouping}"
            )
        return self


class CiSpec(BaseModel):
    """CI fn shape (per-site, ``LayerwiseTransformerCiFn``)."""

    model_config = ConfigDict(extra="forbid")
    d: PositiveInt
    n_blocks: PositiveInt
    n_heads: PositiveInt | None = None
    mlp_hidden: PositiveInt | None = None

    def resolved(self) -> "CiSpec":
        """Return a copy with default-derived n_heads / mlp_hidden filled in."""
        return CiSpec(
            d=self.d,
            n_blocks=self.n_blocks,
            n_heads=self.n_heads if self.n_heads is not None else max(4, self.d // 64),
            mlp_hidden=self.mlp_hidden if self.mlp_hidden is not None else 4 * self.d,
        )


class SweepPoint(BaseModel):
    """One concrete sweep point — every field required, no auto-fill."""

    model_config = ConfigDict(extra="forbid")
    batch: PositiveInt
    seq: PositiveInt
    steps: PositiveInt
    seed: int = 0
    ci: CiSpec
    topology: TopologySpec


class _Defaults(BaseModel):
    """Partial overrides for grid points; same fields as :class:`SweepPoint`, all optional."""

    model_config = ConfigDict(extra="forbid")
    batch: PositiveInt | None = None
    seq: PositiveInt | None = None
    steps: PositiveInt | None = None
    seed: int | None = None
    ci: CiSpec | None = None
    topology: TopologySpec | None = None


class SweepConfig(BaseModel):
    """Top-level sweep YAML schema."""

    model_config = ConfigDict(extra="forbid")
    name_prefix: str
    model: ModelSpec
    runtime: RuntimeSpec
    defaults: _Defaults = _Defaults()
    # Each key → list of values that override defaults. Cartesian product of all
    # keys defines the sweep points. Empty grid → one point from defaults.
    grid: dict[str, list[Any]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_grid_keys(self) -> "SweepConfig":
        allowed = {"batch", "seq", "steps", "seed", "ci", "topology"}
        unknown = set(self.grid) - allowed
        assert not unknown, f"unknown grid keys: {sorted(unknown)} (allowed: {sorted(allowed)})"
        return self

    def expand(self) -> list[SweepPoint]:
        """Cartesian expansion. Each grid axis overrides its same-named default."""
        axes = list(self.grid.items())
        if not axes:
            # No grid: single point from defaults; required fields enforced by SweepPoint.
            return [self._merge_defaults({})]
        keys = [k for k, _ in axes]
        value_lists = [v for _, v in axes]
        points: list[SweepPoint] = []
        for combo in itertools.product(*value_lists):
            overrides = dict(zip(keys, combo, strict=True))
            points.append(self._merge_defaults(overrides))
        return points

    def _merge_defaults(self, overrides: dict[str, Any]) -> SweepPoint:
        merged: dict[str, Any] = {}
        defaults_dump = self.defaults.model_dump(exclude_none=True)
        merged.update(defaults_dump)
        merged.update(overrides)
        return SweepPoint.model_validate(merged)
