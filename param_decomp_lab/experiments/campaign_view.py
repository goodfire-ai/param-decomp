"""`pd-campaign-view <group>` — save the standard wandb workspace view for a campaign.

A campaign is a coordinated multi-run matrix sharing one wandb `group` (set at launch
via `--group`). Each campaign used to hand-build its saved workspace view; this CLI
codes the standard instead, so the view layout evolves by PR. It creates a saved view
named `campaign <group>` in the target project, runset-filtered to the group, with
sections mirroring the canonical metric namespaces (`param_decomp.run` documents the
tree).

Panels list the config-stable keys; run-shaped families (per-site `eval/l0/*`, per-site
grad-norm leaves, campaign-specific recon terms beyond the common set) stay in the
UI's hidden-panels pool — pull the ones a campaign cares about into its sections by
hand after seeding. Re-running always creates a NEW view (the wandb API exposes no
list-views surface to update by name); delete stale ones in the UI.
"""

import fire
import wandb_workspaces.reports.v2 as wr
import wandb_workspaces.workspaces as ws

from param_decomp_lab.infra.wandb import DEFAULT_WANDB_PROJECT, get_wandb_entity

# Masked-forward variants of the fast CE/KL eval (`param_decomp.eval`); `zero_masked`
# has no `ce_difference_*` key.
_CE_KL_VARIANTS = (
    "ci_masked",
    "stoch_masked",
    "unmasked",
    "random_masked",
    "rounded_masked",
    "zero_masked",
)

_REFEREE_CONVENTION = (
    "**Reading the referee**: `eval/loss/PGDReconLoss` is noisy per-eval — compare arms "
    "on last-5/10-eval **medians**, and prefer mid-window reads (e.g. step 35k of 40k) "
    "over endpoints."
)


def _plot(title: str, keys: tuple[str, ...], log_y: bool = False) -> wr.LinePlot:
    return wr.LinePlot(title=title, x="Step", y=list(keys), log_y=log_y or None)


def _campaign_sections() -> list[ws.Section]:
    referee = ws.Section(
        name="Referee",
        panels=[
            _plot("PGD recon (the referee)", ("eval/loss/PGDReconLoss",), log_y=True),
            _plot("KL (CI-masked)", ("eval/ce_kl/kl_ci_masked",), log_y=True),
            _plot("CE difference (CI-masked)", ("eval/ce_kl/ce_difference_ci_masked",)),
            wr.MarkdownPanel(markdown=_REFEREE_CONVENTION),
        ],
        is_open=True,
    )
    recon_quality = ws.Section(
        name="Recon quality (ce_kl)",
        panels=[
            _plot(
                "KL by masking variant",
                tuple(f"eval/ce_kl/kl_{v}" for v in _CE_KL_VARIANTS),
                log_y=True,
            ),
            _plot(
                "CE difference by masking variant",
                tuple(
                    f"eval/ce_kl/ce_difference_{v}" for v in _CE_KL_VARIANTS if v != "zero_masked"
                ),
            ),
        ],
        is_open=True,
    )
    train_losses = ws.Section(
        name="Train losses",
        panels=[
            _plot("total", ("train/loss/total",), log_y=True),
            _plot(
                "faith + imp-min",
                (
                    "train/loss/FaithfulnessLoss",
                    "train/loss/ImportanceMinimalityLoss",
                    "train/loss/SmoothL0ImportanceMinimalityLoss",
                    "train/loss/FrequencyMinimalityLoss",
                ),
                log_y=True,
            ),
            _plot(
                "recon terms (common set)",
                (
                    "train/loss/ChunkwiseSubsetReconLoss",
                    "train/loss/PersistentPGDReconLoss",
                    "train/loss/StochasticReconSubsetLoss",
                    "train/loss/PGDReconLoss",
                ),
                log_y=True,
            ),
        ],
    )
    health = ws.Section(
        name="Health",
        panels=[
            _plot(
                "grad-norm summaries (median)",
                (
                    "train/grad_norms/summary/components/median",
                    "train/grad_norms/summary/ci_fns/median",
                    "train/grad_norms/summary/total/median",
                ),
                log_y=True,
            ),
            _plot("peak memory (GB/rank)", ("train/mem/peak_gb_per_rank",)),
            _plot("step time (s)", ("train/perf/step_time_s",)),
        ],
    )
    schedules = ws.Section(
        name="Schedules",
        panels=[
            _plot(
                "learning rates",
                (
                    "train/schedules/lr/components",
                    "train/schedules/lr/ci_fn",
                    "train/schedules/lr/src",
                ),
                log_y=True,
            ),
            _plot("imp-min anneal", ("train/schedules/p_imp", "train/schedules/gamma_imp")),
        ],
    )
    return [referee, recon_quality, train_losses, health, schedules]


def make_campaign_view(group: str, project: str, entity: str) -> ws.Workspace:
    """Pure builder: the standard campaign workspace, runset-filtered to `group`."""
    return ws.Workspace(
        entity=entity,
        project=project,
        name=f"campaign {group}",
        sections=_campaign_sections(),
        settings=ws.WorkspaceSettings(max_runs=50),
        runset_settings=ws.RunsetSettings(filters=[ws.Metric("Group") == group]),
    )


def main(group: str, project: str = DEFAULT_WANDB_PROJECT, entity: str | None = None) -> None:
    workspace = make_campaign_view(group, project, entity or get_wandb_entity())
    workspace.save_as_new_view()
    print(f"saved campaign view for group {group!r}: {workspace.url}", flush=True)


def cli() -> None:
    fire.Fire(main)
