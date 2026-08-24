#!/usr/bin/env python3
"""Derive a validated LM profiling config from one canonical full config."""

import argparse
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import yaml

from param_decomp.experiments.lm.config import LMExperimentConfig

ShardingPreset = Literal["owner", "zero1", "ddp"]


@dataclass(frozen=True)
class ProfileShape:
    layers: int
    batch_size: int
    replicate: int
    fsdp: int
    tp: int
    steps: int
    profile_steps: int
    sharding: ShardingPreset

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if name != "sharding":
                assert isinstance(value, int) and value > 0, (name, value)
        assert self.profile_steps <= self.steps, (self.profile_steps, self.steps)

    @property
    def run_name(self) -> str:
        return (
            f"profile-h100-{self.layers}l-b{self.batch_size}-"
            f"r{self.replicate}-f{self.fsdp}-t{self.tp}-"
            f"semantic-{self.sharding}-adam"
        )


def _mapping(value: object, path: str) -> dict[str, Any]:
    assert isinstance(value, dict), f"{path} must be a mapping"
    return cast(dict[str, Any], value)


def derive_profile(raw: dict[str, Any], shape: ProfileShape) -> dict[str, Any]:
    LMExperimentConfig.model_validate(raw)
    derived = copy.deepcopy(raw)
    decomposition = _mapping(derived["decomposition"], "decomposition")
    sites = _mapping(decomposition["sites"], "decomposition.sites")
    layers = _mapping(sites["layers"], "decomposition.sites.layers")
    assert layers["kind"] == "range", "profile derivation requires a contiguous layer range"
    assert layers["start"] == 0, "profile derivation currently requires a zero-based layer range"
    layers["end"] = shape.layers

    pd = _mapping(derived["pd"], "pd")
    pd["batch_size"] = shape.batch_size
    pd["steps"] = shape.steps
    losses = pd["loss_metrics"]
    assert isinstance(losses, list), "pd.loss_metrics must be a list"
    hidden_reconstructions = [
        _mapping(loss["hidden_acts_reconstruction"], "hidden_acts_reconstruction")
        for loss in losses
        if isinstance(loss, dict) and loss.get("hidden_acts_reconstruction") is not None
    ]
    assert len(hidden_reconstructions) == 1, (
        "canonical profiling config must declare exactly one hidden-acts reconstruction"
    )
    hidden_reconstructions[0]["points"] = [f"resid.{layer}" for layer in range(1, shape.layers + 1)]

    runtime = _mapping(derived["runtime"], "runtime")
    runtime["replicate"] = shape.replicate
    runtime["fsdp"] = shape.fsdp
    runtime["tp"] = shape.tp
    runtime["sharding"] = shape.sharding
    runtime["profiling"] = {"kind": "ad_hoc", "steps": shape.profile_steps}

    derived["run_name"] = shape.run_name
    LMExperimentConfig.model_validate(derived)
    return derived


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--layers", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--replicate", type=int, required=True)
    parser.add_argument("--fsdp", type=int, required=True)
    parser.add_argument("--tp", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--profile-steps", type=int, required=True)
    parser.add_argument("--sharding", choices=("owner", "zero1", "ddp"), required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    assert not args.output.exists(), f"refusing to overwrite {args.output}"
    raw = yaml.safe_load(args.base.read_text())
    shape = ProfileShape(
        layers=args.layers,
        batch_size=args.batch_size,
        replicate=args.replicate,
        fsdp=args.fsdp,
        tp=args.tp,
        steps=args.steps,
        profile_steps=args.profile_steps,
        sharding=args.sharding,
    )
    derived = derive_profile(_mapping(raw, str(args.base)), shape)
    args.output.write_text(yaml.safe_dump(derived, sort_keys=False))


if __name__ == "__main__":
    main()
