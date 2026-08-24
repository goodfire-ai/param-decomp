"""The trainer's built-run bundle — the pydantic algorithm config (`PDConfig` / `Cadence`)
plus the objects a composition root resolves/builds before entering the generic engine
(`run.py::run_decomposition_training`).

The engine reads `pd` / `cadence` DIRECTLY (no flattened mirror): optimizers,
faith warmup, loss metrics, seed, steps all come off `PDConfig`. The bundle only
carries the things that are genuinely BUILT lab-side and cannot live in the pydantic schema:
the run identity (`RunInstance`), the decomposed `target` (typed by the `TargetSites`
protocol — just `.sites`), the domain-resolved `data` plan (generic and opaque to core), and the built CI-fn architecture (`CIFnArch`). The YAML→bundle
conversion is composition and lives lab-side
(`param_decomp/experiments/`).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from param_decomp.core.ci_fn import CIFnArch
from param_decomp.core.components import SiteC
from param_decomp.core.configs import (
    Cadence,
    PDConfigBase,
    ResumeProvenance,
    WandbConfig,
)

LAUNCH_CONFIG_FILENAME = "launch_config.yaml"
"""The self-contained run config pinned into each run dir. Not `config.yaml` — that basename
clashes with wandb's own run-config file, which `wandb.save` would clobber via symlink."""


class TargetSites(Protocol):
    """The only thing the generic engine needs from a target config: its decomposed
    sites. LM target configs live lab-side; non-LM targets (the toys) live in the lab
    and supply their own config dataclass — the core never names them."""

    @property
    def sites(self) -> tuple[SiteC, ...]: ...


@dataclass(frozen=True)
class RunInstance:
    """A run's identity + logging lineage — the per-run bits that are NOT algorithm config.
    Minted/stamped by the launcher; a toy mints its own."""

    run_name: str
    """Human-readable display name (the wandb run NAME)."""
    run_id: str
    """Canonical `p-<8hex>` id (wandb run ID + run-dir name)."""
    out_dir: Path
    wandb: WandbConfig | None
    resume_provenance: ResumeProvenance | None
    """Set on a fine-tune run: the parent run dir + step to initialize V/U + ci_fn from
    (SPEC S33). `None` for a fresh-from-init run."""

    @property
    def run_dir(self) -> Path:
        return self.out_dir / self.run_id


@dataclass(frozen=True)
class BuiltRun[DataT, TargetT: TargetSites, PDT: PDConfigBase]:
    """Everything the generic engine needs for one decomposition run: the pydantic
    algorithm config (read DIRECTLY) plus the lab-built objects.

    `pd` / `cadence` are the verbatim pydantic config — the engine reads
    optimizers / loss metrics / faith warmup / seed / steps / cadence off them,
    so there is no flattened mirror to drift. The rest is what composition resolves: the
    run identity, the decomposed target (only `.sites` is read), opaque domain data, and the built CI-fn architecture.

    The compute substrate is deliberately NOT here — core accepts no config type for it at
    all, and the section is declared by the one domain that has one. Its values do reach
    core, but always unpacked into primitives — explicit mesh axes into `sharding`, the
    placement spec into `placement`, the remat flags and `compiler_options` into the
    engine — each passed by the composition root that owns the substrate. So a domain that
    has none (the single-device toys) carries none, rather than carrying a field it must
    fill with lies."""

    pd: PDT
    cadence: Cadence
    run: RunInstance
    target: TargetT
    """The decomposed target config — only its `sites` are read by the generic engine. An
    LM target config (lab-side) or a lab toy target config (satisfying `TargetSites`)."""
    data: DataT
    """Domain data plan; `None` when composition supplies batches directly."""
    ci_fn: CIFnArch
