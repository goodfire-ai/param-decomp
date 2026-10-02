"""Shared config schema for in-repo experiment YAMLs, plus the shared run-identity
helpers every experiment reuses.

Each experiment subclasses `ExperimentConfig` to fix the concrete `target` / `data` types.
The generic engine reads the pydantic `pd` / `cadence` DIRECTLY, so there is no
flattened mirror to build; `run_instance` / `ci_fn_arch` resolve the run identity and the
CI-fn architecture, and each experiment's `run.py` assembles the rest (target + data).
"""

import re
from pathlib import Path

from param_decomp.core.base_config import BaseConfig
from param_decomp.core.built_run import RunInstance
from param_decomp.core.configs import (
    Cadence,
    PDConfig,
    ResumeProvenance,
    WandbConfig,
)
from param_decomp.core.run_files import LAUNCH_CONFIG_FILENAME


class ExperimentConfigBase(BaseConfig):
    """The domain-AGNOSTIC sections of an in-repo experiment YAML.

    The domain-specific, co-varying config — `target`, the `decomposition` apparatus
    (site-spec + CI-fn arch), and `data` — is declared as CONCRETE fields on each per-domain
    subclass (`LMExperimentConfig`, `TMSExperimentConfig`, …), NOT as generic type params:
    those three vary together as one axis (the domain), so binding them on the concrete
    subclass makes a cross-domain mismatch (e.g. LM target + toy CI fn) unrepresentable.

    The compute substrate (`runtime:` — world size, placement, remat, XLA flags, the
    pre-process env) is per-domain for the same reason: the LM declares one, the toys
    declare nothing, so a `runtime:` block in a toy YAML is unrepresentable rather than
    refused field-by-field by a validator that has to be kept honest as the substrate grows.

    Each domain owns its `eval:` schema. Omit `wandb:` to skip wandb (the run still writes
    `launch_config.yaml` + checkpoints locally).

    The run id is NOT a config field: the entry point mints it or accepts it explicitly,
    then passes it to `run_instance`. The run dir is a pure function of the
    entry point's `data_root` + id (`<data_root>/runs/<run_id>`).
    """

    run_name: str
    """Human-readable display name (the wandb run NAME)."""

    cadence: Cadence
    wandb: WandbConfig | None = None


class ExperimentConfig(ExperimentConfigBase):
    """The plain-VPD run shape's shared base: the identity/rhythm sections + the plain
    algorithm config. A targeted (tPD) run shape subclasses `ExperimentConfigBase`
    directly and declares `pd: TargetedPDConfig` — the run shapes are siblings, never a
    mode of one another."""

    pd: PDConfig


_RUN_ID_PATTERN = re.compile(r"^p-[0-9a-f]{8}$")


def run_instance(
    cfg: ExperimentConfigBase,
    run_id: str,
    data_root: Path,
    resume_provenance: ResumeProvenance | None,
) -> RunInstance:
    """The resolved run identity + logging lineage. The composition root supplies
    `run_id` (a toy mints its own); the run dir is `<data_root>/runs/<run_id>`. Fine-tune provenance is
    explicit because only domains that implement resume expose it."""
    assert _RUN_ID_PATTERN.match(run_id), f"run_id must be p-<8hex>, got {run_id!r}"
    return RunInstance(
        run_name=cfg.run_name,
        run_id=run_id,
        out_dir=data_root / "runs",
        wandb=cfg.wandb,
        resume_provenance=resume_provenance,
    )


def apply_wandb_cli_overrides(
    schema_raw: dict[str, object], group: str | None, tags: str | tuple[str, ...] | None
) -> None:
    """Merge the `--group` / `--tags` CLI flags into the raw schema's `wandb:` block
    (in place, pre-validation) — shared by every toy module main."""
    if group is None and tags is None:
        return
    raw_wandb = schema_raw.get("wandb")
    wandb_cfg: dict[str, object] = dict(raw_wandb) if isinstance(raw_wandb, dict) else {}
    if group is not None:
        wandb_cfg["group"] = group
    if tags is not None:
        # Fire parses a comma-separated `--tags a,b,c` into a tuple, but keeps a value
        # with a hyphen (e.g. `a,b,c-d`) as a string — normalize both to a list.
        wandb_cfg["tags"] = (
            [s.strip() for s in tags.split(",") if s.strip()]
            if isinstance(tags, str)
            else [str(t).strip() for t in tags]
        )
    schema_raw["wandb"] = wandb_cfg


def pin_launch_config(run_dir: Path, resolved_yaml: str) -> None:
    """First run pins the resolved launch config into the run dir; a rerun with the same
    run id byte-compares against the pin (resuming must not change the config)."""
    pinned = run_dir / LAUNCH_CONFIG_FILENAME
    if pinned.exists():
        assert pinned.read_text() == resolved_yaml, (
            f"{pinned} differs from the invoked config — refusing to resume with a changed config"
        )
    else:
        pinned.write_text(resolved_yaml)
