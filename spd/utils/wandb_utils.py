import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import wandb
import wandb.errors
import wandb_workspaces.reports.v2 as wr
import wandb_workspaces.workspaces as ws
from dotenv import load_dotenv
from wandb.apis.public import File, Run

from spd.base_config import BaseConfig
from spd.log import logger
from spd.registry import EXPERIMENT_REGISTRY
from spd.settings import REPO_ROOT
from spd.utils.general_utils import fetch_latest_checkpoint_name

WORKSPACE_TEMPLATES = {
    "default": "https://wandb.ai/goodfire/spd?nw=css034maye",
    "tms_5-2": "https://wandb.ai/goodfire/spd?nw=css034maye",
    "tms_40-10": "https://wandb.ai/goodfire/spd?nw=css034maye",
    "tms_5-2-id": "https://wandb.ai/goodfire/nathu-spd-test-proj?nw=iytdd13y9d0",
    "tms_40-10-id": "https://wandb.ai/goodfire/nathu-spd-test-proj?nw=iytdd13y9d0",
    "resid_mlp1": "https://wandb.ai/goodfire/nathu-spd?nw=5im20fd95rg",
    "resid_mlp2": "https://wandb.ai/goodfire/nathu-spd?nw=5im20fd95rg",
    "resid_mlp3": "https://wandb.ai/goodfire/nathu-spd?nw=5im20fd95rg",
}

# Regex patterns for parsing W&B run references
# Run IDs can be 8 chars (e.g., "d2ec3bfe") or prefixed with char-dash (e.g., "s-d2ec3bfe")
DEFAULT_WANDB_ENTITY = "goodfire"
DEFAULT_WANDB_PROJECT = "spd"

_RUN_ID_PATTERN = r"(?:[a-z0-9]-)?[a-z0-9]{8}"
_BARE_RUN_ID_RE = re.compile(r"^(s-[a-z0-9]{8})$")
_WANDB_PATH_RE = re.compile(rf"^([^/\s]+)/([^/\s]+)/({_RUN_ID_PATTERN})$")
_WANDB_PATH_WITH_RUNS_RE = re.compile(rf"^([^/\s]+)/([^/\s]+)/runs/({_RUN_ID_PATTERN})$")
_WANDB_URL_RE = re.compile(
    rf"^https://wandb\.ai/([^/]+)/([^/]+)/runs/({_RUN_ID_PATTERN})(?:/[^?]*)?(?:\?.*)?$"
)

# Short names for metric classes, used for W&B run names and view names
METRIC_CONFIG_SHORT_NAMES: dict[str, str] = {
    # Loss metrics
    "FaithfulnessLoss": "Faith",
    "ImportanceMinimalityLoss": "ImpMin",
    "StochasticReconLoss": "StochRecon",
    "StochasticReconSubsetLoss": "StochReconSub",
    "StochasticReconLayerwiseLoss": "StochReconLayer",
    "CIMaskedReconLoss": "CIMaskRecon",
    "CIMaskedReconSubsetLoss": "CIMaskReconSub",
    "CIMaskedReconLayerwiseLoss": "CIMaskReconLayer",
    "PGDReconLoss": "PGDRecon",
    "PGDReconSubsetLoss": "PGDReconSub",
    "PGDReconLayerwiseLoss": "PGDReconLayer",
    "PersistentPGDReconLoss": "PersistPGDRecon",
    "PersistentPGDReconSubsetLoss": "PersistPGDReconSub",
    "StochasticHiddenActsReconLoss": "StochHiddenActRecon",
    "CIHiddenActsReconLoss": "CIHiddenActRecon",
    "StochasticAttnPatternsReconLoss": "StochAttnRecon",
    "CIMaskedAttnPatternsReconLoss": "CIAttnRecon",
    "UnmaskedReconLoss": "UnmaskedRecon",
    # Eval metrics
    "CEandKLLosses": "CEandKL",
    "CIHistograms": "CIHist",
    "CI_L0": "CI_L0",
    "CIMeanPerComponent": "CIMeanPerComp",
    "ComponentActivationDensity": "CompActDens",
    "IdentityCIError": "IdCIErr",
    "PermutedCIPlots": "PermCIPlots",
    "UVPlots": "UVPlots",
    "StochasticReconSubsetCEAndKL": "StochReconSubCEKL",
    "PGDMultiBatchReconLoss": "PGDMultiBatchRecon",
    "PGDMultiBatchReconSubsetLoss": "PGDMultiBatchReconSub",
    "PersistentPGDReconEval": "PersistPGDReconEval",
    "PersistentPGDReconSubsetEval": "PersistPGDReconSubEval",
}


def get_wandb_entity() -> str:
    """Get the WandB entity from env var or the authenticated user's default entity."""
    load_dotenv(override=True)
    entity = os.getenv("WANDB_ENTITY")
    if entity is None:
        entity = wandb.Api().default_entity
    assert entity is not None, (
        "Could not determine WandB entity. Set WANDB_ENTITY in .env or log in with `wandb login`."
    )
    return entity


def get_wandb_run_url(project: str, run_id: str) -> str:
    """Get the direct WandB URL for a run."""
    return f"https://wandb.ai/{get_wandb_entity()}/{project}/runs/{run_id}"


def wandb_path_to_url(wandb_path: str) -> str:
    """Convert a WandB run path to a URL."""
    entity, project, run_id = parse_wandb_run_path(wandb_path)
    return f"https://wandb.ai/{entity}/{project}/runs/{run_id}"


def _parse_metric_config_key(key: str) -> tuple[str, str, str] | None:
    """Parse a metric config key into (list_field, classname, param).

    Args:
        key: Flattened key like "loss_metric_configs.ImportanceMinimalityLoss.pnorm"

    Returns:
        Tuple of (list_field, classname, param) if it's a metric config key, None otherwise
    """
    parts = key.split(".")
    if len(parts) >= 3 and parts[0] in ("loss_metric_configs", "eval_metric_configs"):
        list_field = parts[0]
        classname = parts[1]
        param = ".".join(parts[2:])  # Handle nested params like "task_config.feature_probability"
        return (list_field, classname, param)
    return None


def generate_wandb_run_name(params: dict[str, Any]) -> str:
    """Generate a W&B run name based on sweep parameters.

    Handles special formatting for metric configs (loss_metric_configs, eval_metric_configs)
    by abbreviating classnames and grouping parameters by metric type.

    Args:
        params: Dictionary of flattened sweep parameters

    Returns:
        Formatted run name string

    Example:
        >>> params = {
        ...     "seed": 42,
        ...     "loss_metric_configs.ImportanceMinimalityLoss.pnorm": 0.9,
        ...     "loss_metric_configs.ImportanceMinimalityLoss.coeff": 0.001,
        ... }
        >>> generate_wandb_run_name(params)
        "seed-42-ImpMin-coeff-0.001-pnorm-0.9"
    """
    # Group parameters by type: regular params and metric config params
    regular_params: list[tuple[str, Any]] = []
    metric_params: dict[str, list[tuple[str, Any]]] = {}  # classname -> [(param, value), ...]

    for key, value in params.items():
        parsed = _parse_metric_config_key(key)
        if parsed:
            _, classname, param = parsed
            # Get short name for the classname
            short_name = METRIC_CONFIG_SHORT_NAMES.get(classname, classname)
            if short_name not in metric_params:
                metric_params[short_name] = []
            metric_params[short_name].append((param, value))
        else:
            regular_params.append((key, value))

    # Build parts list
    parts: list[str] = []

    # Add regular params (sorted for consistency)
    for key, value in sorted(regular_params):
        parts.append(f"{key}-{value}")

    # Add metric config params (sorted by classname, then by param)
    for short_name in sorted(metric_params.keys()):
        parts.append(short_name)
        for param, value in sorted(metric_params[short_name]):
            parts.append(f"{param}-{value}")

    return "-".join(parts)


def parse_wandb_run_path(input_path: str) -> tuple[str, str, str]:
    """Parse various W&B run reference formats into (entity, project, run_id).

    Accepts:
    - "s-xxxxxxxx" (bare SPD run ID, assumes goodfire/spd)
    - "entity/project/runId" (compact form)
    - "entity/project/runs/runId" (with /runs/)
    - "wandb:entity/project/runId" (with wandb: prefix)
    - "wandb:entity/project/runs/runId" (full wandb: form)
    - "https://wandb.ai/entity/project/runs/runId..." (URL)

    Returns:
        Tuple of (entity, project, run_id)

    Raises:
        ValueError: If the input doesn't match any expected format.
    """
    s = input_path.strip()

    # Strip wandb: prefix if present
    if s.startswith("wandb:"):
        s = s[6:]

    # Bare run ID (e.g. "s-17805b61") → default entity/project
    if m := _BARE_RUN_ID_RE.match(s):
        return DEFAULT_WANDB_ENTITY, DEFAULT_WANDB_PROJECT, m.group(1)

    # Try compact form: entity/project/runid
    if m := _WANDB_PATH_RE.match(s):
        return m.group(1), m.group(2), m.group(3)

    # Try form with /runs/: entity/project/runs/runid
    if m := _WANDB_PATH_WITH_RUNS_RE.match(s):
        return m.group(1), m.group(2), m.group(3)

    # Try full URL
    if m := _WANDB_URL_RE.match(s):
        return m.group(1), m.group(2), m.group(3)

    raise ValueError(
        f"Invalid W&B run reference. Expected one of:\n"
        f' - "s-xxxxxxxx" (bare run ID)\n'
        f' - "entity/project/xxxxxxxx"\n'
        f' - "entity/project/runs/xxxxxxxx"\n'
        f' - "wandb:entity/project/runs/xxxxxxxx"\n'
        f' - "https://wandb.ai/entity/project/runs/xxxxxxxx"\n'
        f'Got: "{input_path}"'
    )


def flatten_metric_configs(config_dict: dict[str, Any]) -> dict[str, Any]:
    """Flatten loss_metric_configs and eval_metric_configs into dot-notation for wandb searchability.

    Converts:
        loss_metric_configs: [{"classname": "ImportanceMinimalityLoss", "coeff": 0.1, "pnorm": 1.0}]
    To:
        loss_metric_configs.ImportanceMinimalityLoss.coeff: 0.1
        loss_metric_configs.ImportanceMinimalityLoss.pnorm: 1.0
    """
    flattened: dict[str, Any] = {}

    for config_list_name in ["loss_metric_configs", "eval_metric_configs"]:
        if config_list_name not in config_dict:
            continue

        configs = config_dict[config_list_name]
        assert isinstance(configs, list), f"{config_list_name} should be a list"

        for config_item in configs:
            assert isinstance(config_item, dict), f"{config_list_name} should have dicts"

            classname = config_item["classname"]
            assert isinstance(classname, str), f"{config_list_name} should have a classname"
            short_name = METRIC_CONFIG_SHORT_NAMES[classname]

            for key, value in config_item.items():
                if key == "classname":
                    continue
                # Get a "loss" or "eval" prefix
                prefix = config_list_name.split("_")[0]
                # Create flattened key
                flat_key = f"{prefix}.{short_name}.{key}"
                flattened[flat_key] = value

    return flattened


def fetch_latest_wandb_checkpoint(run: Run, prefix: str | None = None) -> File:
    """Fetch the latest checkpoint from a wandb run."""
    filenames = [file.name for file in run.files() if file.name.endswith((".pth", ".pt"))]
    latest_checkpoint_name = fetch_latest_checkpoint_name(filenames, prefix)
    latest_checkpoint_remote = run.file(latest_checkpoint_name)
    return latest_checkpoint_remote


def fetch_wandb_run_dir(run_id: str) -> Path:
    """Find or create a directory in the W&B cache for a given run.

    We first check if we already have a directory with the suffix "run_id" (if we created the run
    ourselves, a directory of the name "run-<timestamp>-<run_id>" should exist). If not, we create a
    new wandb_run_dir.
    """
    # Default to REPO_ROOT/wandb
    base_cache_dir = REPO_ROOT / "wandb"
    base_cache_dir.mkdir(parents=True, exist_ok=True)

    # Set default wandb_run_dir
    wandb_run_dir = base_cache_dir / run_id / "files"

    # Check if we already have a directory with the suffix "run_id"
    presaved_run_dirs = [
        d for d in base_cache_dir.iterdir() if d.is_dir() and d.name.endswith(run_id)
    ]
    # If there is more than one dir, just ignore the presaved dirs and use the new wandb_run_dir
    if presaved_run_dirs and len(presaved_run_dirs) == 1:
        presaved_file_path = presaved_run_dirs[0] / "files"
        if presaved_file_path.exists():
            # Found a cached run directory, use it
            wandb_run_dir = presaved_file_path

    wandb_run_dir.mkdir(parents=True, exist_ok=True)
    return wandb_run_dir


def download_wandb_file(run: Run, wandb_run_dir: Path, file_name: str) -> Path:
    """Download a file from W&B. Don't overwrite the file if it already exists.

    Args:
        run: The W&B run to download from
        file_name: Name of the file to download
        wandb_run_dir: The directory to download the file to
    Returns:
        Path to the downloaded file
    """
    file_on_wandb = run.file(file_name)
    assert isinstance(file_on_wandb, File)
    file_on_wandb.download(exist_ok=True, replace=False, root=str(wandb_run_dir))
    return wandb_run_dir / file_name


def init_wandb(
    config: BaseConfig,
    project: str,
    run_id: str,
    name: str | None = None,
    tags: list[str] | None = None,
) -> None:
    """Initialize Weights & Biases and log the config.

    Args:
        config: The config to log.
        project: The name of the wandb project.
        run_id: The unique run ID (from ExecutionStamp).
        name: The name of the wandb run.
        tags: Optional list of tags to add to the run.
    """
    wandb.init(
        id=run_id,
        project=project,
        entity=get_wandb_entity(),
        name=name,
        tags=tags,
    )
    assert wandb.run is not None
    wandb.run.log_code(
        root=str(REPO_ROOT / "spd"), exclude_fn=lambda path: "out" in Path(path).parts
    )

    config_dict = config.model_dump(mode="json")
    # We also want flattened names for easier wandb searchability
    flattened_config_dict = flatten_metric_configs(config_dict)
    # Remove the nested metric configs to avoid duplication (if they exist)
    if "loss_metric_configs" in config_dict:
        del config_dict["loss_metric_configs"]
    if "eval_metric_configs" in config_dict:
        del config_dict["eval_metric_configs"]
    wandb.config.update({**config_dict, **flattened_config_dict})


def ensure_project_exists(project: str) -> None:
    """Ensure the W&B project exists by creating a dummy run if needed."""
    api = wandb.Api()

    # Check if project exists in the list of projects
    if project not in [p.name for p in api.projects()]:
        # Project doesn't exist, create it with a dummy run
        logger.info(f"Creating W&B project '{project}'...")
        run = wandb.init(project=project, name="project_init", tags=["init"])
        run.finish()
        logger.info(f"Project '{project}' created successfully")


def create_workspace_view(launch_id: str, experiment_name: str, project: str) -> str:
    """Create a wandb workspace view for an experiment."""
    # Use experiment-specific template if available
    template_url: str = WORKSPACE_TEMPLATES.get(experiment_name, WORKSPACE_TEMPLATES["default"])
    workspace: ws.Workspace = ws.Workspace.from_url(template_url)

    # Override the project to match what we're actually using
    workspace.project = project

    # Update the workspace name
    workspace.name = f"{experiment_name} - {launch_id}"

    # Filter for runs that have BOTH the launch_id AND experiment name tags
    workspace.runset_settings.filters = [
        ws.Tags("tags").isin([launch_id]),
        ws.Tags("tags").isin([experiment_name]),
    ]

    # Save as a new view
    workspace.save_as_new_view()

    return workspace.url


def create_wandb_report(
    report_title: str,
    launch_id: str,
    branch_name: str,
    commit_hash: str | None,
    experiments: list[str],
    include_run_comparer: bool,
    project: str,
    report_total_width: int = 24,
) -> str:
    """Create a W&B report for the launch."""
    report = wr.Report(
        project=project,
        title=report_title,
        description=f"Experiments: {', '.join(experiments)}",
        width="fluid",
    )

    report.blocks.append(
        wr.MarkdownBlock(text=f"Branch: `{branch_name}`\nCommit: `{commit_hash or 'none'}`")
    )

    # Create separate panel grids for each experiment
    for experiment in experiments:
        task_name: str = EXPERIMENT_REGISTRY[experiment].task_name

        # Use launch_id and experiment name tags for filtering
        combined_filter = (
            f'(Tags("tags") in ["{launch_id}"]) and (Tags("tags") in ["{experiment}"])'
        )

        # Create runset for this specific experiment
        runset = wr.Runset(
            name=f"{experiment} Runs",
            filters=combined_filter,
        )

        # Build panels list
        panels: list[wr.interface.PanelTypes] = []
        y = 0

        if task_name in ["tms", "resid_mlp"]:
            ci_height = 12
            panels.append(
                wr.MediaBrowser(
                    media_keys=["eval/figures/causal_importances_upper_leaky"],
                    layout=wr.Layout(x=0, y=0, w=report_total_width, h=ci_height),
                    num_columns=6,
                )
            )
            y += ci_height

        loss_plots_height = 6
        loss_plots = [
            ["train/loss/stochastic_recon_layerwise", "train/loss/stochastic_recon"],
            ["eval/loss/faithfulness"],
            ["train/loss/importance_minimality"],
        ]
        for i, y_keys in enumerate(loss_plots):
            loss_plots_width = report_total_width // len(loss_plots)
            x_offset = i * loss_plots_width
            panels.append(
                wr.LinePlot(
                    x="Step",
                    y=y_keys,  # pyright: ignore[reportArgumentType]
                    log_y=True,
                    layout=wr.Layout(x=x_offset, y=y, w=loss_plots_width, h=loss_plots_height),
                )
            )
        y += loss_plots_height

        if task_name in ["tms", "resid_mlp"]:
            # Add target CI error plots
            target_ci_weight = 6
            target_ci_width = report_total_width // 2
            panels.append(
                wr.LinePlot(
                    x="Step",
                    y=["target_solution_error/total"],
                    title="Target CI Error (Tolerance=0.1)",
                    layout=wr.Layout(x=0, y=y, w=target_ci_width, h=target_ci_weight),
                )
            )
            panels.append(
                wr.LinePlot(
                    x="Step",
                    y=["target_solution_error/total_0p2"],
                    title="Target CI Error (Tolerance=0.2)",
                    layout=wr.Layout(x=target_ci_width, y=y, w=target_ci_width, h=target_ci_weight),
                )
            )
            y += target_ci_weight

        # Only add KL loss plots for language model experiments
        if task_name == "lm":
            kl_height = 6
            kl_width = report_total_width // 3
            x_offset = 0
            panels.append(
                wr.LinePlot(
                    x="Step",
                    y=["eval/kl/ci_masked"],
                    layout=wr.Layout(x=x_offset, y=y, w=kl_width, h=kl_height),
                )
            )
            x_offset += kl_width
            panels.append(
                wr.LinePlot(
                    x="Step",
                    y=["eval/kl/unmasked"],
                    layout=wr.Layout(x=x_offset, y=y, w=kl_width, h=kl_height),
                )
            )
            x_offset += kl_width
            panels.append(
                wr.LinePlot(
                    x="Step",
                    y=["eval/kl/stoch_masked"],
                    layout=wr.Layout(x=x_offset, y=y, w=kl_width, h=kl_height),
                )
            )
            x_offset += kl_width
            y += kl_height

            ce_height = 6
            ce_width = report_total_width // 3
            x_offset = 0
            panels.append(
                wr.LinePlot(
                    x="Step",
                    y=["eval/ce_unrecovered/ci_masked"],
                    layout=wr.Layout(x=x_offset, y=y, w=ce_width, h=ce_height),
                )
            )
            x_offset += kl_width
            panels.append(
                wr.LinePlot(
                    x="Step",
                    y=["eval/ce_unrecovered/unmasked"],
                    layout=wr.Layout(x=x_offset, y=y, w=ce_width, h=ce_height),
                )
            )
            x_offset += kl_width
            panels.append(
                wr.LinePlot(
                    x="Step",
                    y=["eval/ce_unrecovered/stoch_masked"],
                    layout=wr.Layout(x=x_offset, y=y, w=ce_width, h=ce_height),
                )
            )
            x_offset += kl_width
            y += ce_height

        if include_run_comparer:
            run_comparer_height = 10
            panels.append(
                wr.RunComparer(
                    diff_only=True,
                    layout=wr.Layout(x=0, y=y, w=report_total_width, h=run_comparer_height),
                )
            )
            y += run_comparer_height

        panel_grid = wr.PanelGrid(
            runsets=[runset],
            panels=panels,
        )

        # Add title block and panel grid
        report.blocks.append(wr.H2(text=experiment))
        report.blocks.append(panel_grid)

    # Save the report and return URL
    report.save()
    return report.url


@dataclass
class ReportCfg:
    """metadata for setting up a wandb view and optionally a report for the run.

    Args:
        snapshot_branch: Git branch name for the snapshot created by this run.
        commit_hash: Commit hash of the snapshot created by this run.
        report_title: Title for the W&B report. If None, will be generated
    """

    branch: str
    commit_hash: str
    report_title: str | None


def create_view_and_report(
    project: str,
    launch_id: str,
    experiments: list[str],
    report_cfg: ReportCfg | None,
) -> None:
    """set up wandb, creating workspace views and optionally creating a report

    Args:
        project: W&B project name
        launch_id: Launch identifier for this group of jobs
        experiments: List of experiment names to create views for
        report_cfg: How to set up a wandb view, and optionally a report for the run, if at all.
    """
    # Ensure the W&B project exists
    ensure_project_exists(project)

    # Create workspace views for each experiment
    logger.section("Creating workspace views...")
    workspace_urls: dict[str, str] = {}
    for experiment in experiments:
        workspace_url = create_workspace_view(launch_id, experiment, project)
        workspace_urls[experiment] = workspace_url

    # Create report if requested
    report_url: str | None = None
    if report_cfg is not None and len(experiments) > 1:
        report_url = create_wandb_report(
            report_title=report_cfg.report_title or f"SPD Launch Report - {launch_id}",
            launch_id=launch_id,
            branch_name=report_cfg.branch,
            commit_hash=report_cfg.commit_hash,
            experiments=experiments,
            include_run_comparer=True,
            project=project,
        )

    # Print clean summary after wandb messages
    logger.values(
        msg="workspace urls per experiment",
        data={
            **workspace_urls,
            **({"Aggregated Report": report_url} if report_url else {}),
        },
    )


_n_try_wandb_comm_errors = 0


def try_wandb[**P, T](wandb_fn: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T | None:
    """Attempts to call `wandb_fn` and if it fails with a wandb CommError, logs a warning and returns
    None. The choice of wandb CommError is to catch issues communicating with the wandb server but
    not legitimate logging errors, for example not passing a dict to wandb.log, or the wrong
    arguments to wandb.save."""
    global _n_try_wandb_comm_errors
    try:
        return wandb_fn(*args, **kwargs)
    except wandb.errors.CommError as e:
        _n_try_wandb_comm_errors += 1
        logger.error(
            f"wandb communication error, skipping log (total comm errors: {_n_try_wandb_comm_errors}): {e}"
        )
