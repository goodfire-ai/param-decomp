"""Utilities for managing experiment run directories and IDs."""

import copy
import itertools
import json
import os
import secrets
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Final, Literal, NamedTuple

import torch
import yaml

from spd.configs import Config
from spd.log import logger
from spd.settings import SPD_OUT_DIR
from spd.utils.git_utils import (
    create_git_snapshot,
    repo_current_branch,
    repo_current_commit_hash,
    repo_is_clean,
)

# Fields that use discriminated union merging: field_name -> discriminator_field
_DISCRIMINATED_LIST_FIELDS: dict[str, str] = {
    "loss_metric_configs": "classname",
    "eval_metric_configs": "classname",
}


def _save_json(data: Any, path: Path | str, **kwargs: Any) -> None:
    with open(path, "w") as f:
        json.dump(data, f, **kwargs)


def _save_yaml(data: Any, path: Path | str, **kwargs: Any) -> None:
    with open(path, "w") as f:
        yaml.dump(data, f, sort_keys=False, **kwargs)


def _save_torch(data: Any, path: Path | str, **kwargs: Any) -> None:
    torch.save(data, path, **kwargs)


def _save_text(data: str, path: Path | str, encoding: str = "utf-8") -> None:
    with open(path, "w", encoding=encoding) as f:
        f.write(data)


def save_file(data: dict[str, Any] | Any, path: Path | str, **kwargs: Any) -> None:
    """Save a file.

    NOTE: This function was originally designed to save files with specific permissions,
    bypassing the system's umask. This is not needed anymore, but we're keeping this
    abstraction for convenience and brevity.

    File type is determined by extension:
    - .json: Save as JSON
    - .yaml/.yml: Save as YAML
    - .pth/.pt: Save as PyTorch model
    - .txt or other: Save as plain text (data must be string)

    Args:
        data: Data to save (format depends on file type)
        path: File path to save to
        **kwargs: Additional arguments passed to the specific save function
    """
    path = Path(path)
    suffix = path.suffix.lower()

    path.parent.mkdir(parents=True, exist_ok=True)

    if suffix == ".json":
        _save_json(data, path, **kwargs)
    elif suffix in [".yaml", ".yml"]:
        _save_yaml(data, path, **kwargs)
    elif suffix in [".pth", ".pt"]:
        _save_torch(data, path, **kwargs)
    else:
        # Default to text file
        assert isinstance(data, str), f"For {suffix} files, data must be a string, got {type(data)}"
        _save_text(data, path, encoding=kwargs.get("encoding", "utf-8"))


def apply_nested_updates(base_dict: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    """Apply nested updates to a dictionary with flattened keys.

    Supports dot notation for all fields:
        - Regular: "task_config.max_seq_len"
        - Discriminated lists: "loss_metric_configs.Loss1.coeff"

    For discriminated list fields, matches items by discriminator value in the path.
    Preserves base items not mentioned in updates and adds new items from updates.

    Args:
        base_dict: The base configuration dictionary
        updates: Dictionary of flattened key-value pairs

    Returns:
        Updated dictionary (deep copy, original unchanged)
    """
    result = copy.deepcopy(base_dict)

    for key, value in updates.items():
        if "." in key:
            keys = key.split(".")

            # Check if this is a discriminator-based list key
            # Format: "list_field.discriminator_value.field_name..."
            if len(keys) >= 3 and keys[0] in _DISCRIMINATED_LIST_FIELDS:
                list_field = keys[0]
                discriminator_value = keys[1]
                field_path = keys[2:]  # Remaining path after discriminator

                # Ensure the list exists
                if list_field not in result:
                    result[list_field] = []

                if not isinstance(result[list_field], list):
                    raise ValueError(
                        f"Expected '{list_field}' to be a list, got {type(result[list_field])}"
                    )

                # Find or create the item with matching discriminator
                discriminator_field = _DISCRIMINATED_LIST_FIELDS[list_field]
                target_item = None
                for item in result[list_field]:
                    if item.get(discriminator_field) == discriminator_value:
                        target_item = item
                        break

                if target_item is None:
                    # Create new item with discriminator
                    target_item = {discriminator_field: discriminator_value}
                    result[list_field].append(target_item)

                # Navigate the remaining path within the item
                current_item: dict[str, Any] = target_item
                for k in field_path[:-1]:
                    if k not in current_item:
                        current_item[k] = {}
                    assert isinstance(current_item[k], dict)
                    current_item = current_item[k]

                # Set the final value
                current_item[field_path[-1]] = value
            else:
                # Regular dot notation (non-discriminated)
                current: dict[str, Any] = result

                # Navigate to the parent of the final key
                for k in keys[:-1]:
                    if k not in current:
                        current[k] = {}
                    assert isinstance(current[k], dict)
                    current = current[k]

                # Set the final value
                current[keys[-1]] = value
        else:
            # Simple key replacement (no dot notation)
            result[key] = value

    return result


def _extract_value_specs_from_sweep_params(
    obj: Any,
    path: list[str],
    value_specs: list[tuple[str, list[Any]]],
) -> None:
    """Recursively extract all {"values": [...]} specs with flattened paths."""
    if isinstance(obj, dict):
        if "values" in obj and len(obj) == 1:
            # This is a value spec - create flattened key
            flattened_key = ".".join(path)
            value_specs.append((flattened_key, obj["values"]))
        else:
            # Regular dict, recurse
            for key, value in obj.items():
                _extract_value_specs_from_sweep_params(value, path + [key], value_specs)
    elif isinstance(obj, list):
        # All lists must be discriminated
        if len(path) == 0:
            raise ValueError("Cannot have a list at the root level of sweep parameters")

        parent_key = path[-1]
        if parent_key not in _DISCRIMINATED_LIST_FIELDS:
            raise ValueError(
                f"List field '{parent_key}' is not in _DISCRIMINATED_LIST_FIELDS. "
                f"All list fields must be discriminated unions. "
                f"Known discriminated fields: {list(_DISCRIMINATED_LIST_FIELDS.keys())}"
            )

        discriminator_field = _DISCRIMINATED_LIST_FIELDS[parent_key]
        seen_discriminators: set[str] = set()

        for item in obj:
            if not isinstance(item, dict):
                raise ValueError(
                    f"All items in discriminated list '{parent_key}' must be dicts, got {type(item)}"
                )
            if discriminator_field not in item:
                raise ValueError(
                    f"Item in discriminated list '{parent_key}' missing discriminator field '{discriminator_field}': {item}"
                )

            disc_value = item[discriminator_field]
            if not isinstance(disc_value, str):
                raise ValueError(
                    f"Discriminator field '{discriminator_field}' must be a string, got {type(disc_value)}: {disc_value}"
                )

            if disc_value in seen_discriminators:
                raise ValueError(
                    f"Duplicate discriminator value '{disc_value}' in list field '{parent_key}'"
                )
            seen_discriminators.add(disc_value)

            # Recurse into item's fields with discriminator in path
            for field_key, field_value in item.items():
                if field_key == discriminator_field:
                    # Skip the discriminator field - it's already in the path
                    continue
                field_path = path + [disc_value, field_key]
                _extract_value_specs_from_sweep_params(field_value, field_path, value_specs)


def _validate_sweep_params_have_values(
    obj: Any,
    path: list[str],
    parent_list_key: str | None = None,
) -> None:
    """Validate that all leaves have {"values": [...]}, except discriminator fields."""
    if isinstance(obj, dict):
        if "values" in obj:
            return  # This is a value spec
        if not obj:
            return  # Empty dict is ok
        for key, value in obj.items():
            _validate_sweep_params_have_values(value, path + [key], parent_list_key)
    elif isinstance(obj, list):
        # Track that we're inside a discriminated list
        list_field = path[-1] if path else None
        for item in obj:
            _validate_sweep_params_have_values(item, path, parent_list_key=list_field)
    else:
        # Primitive value - check if it's a discriminator field
        if parent_list_key and parent_list_key in _DISCRIMINATED_LIST_FIELDS:
            discriminator_field = _DISCRIMINATED_LIST_FIELDS[parent_list_key]
            if path and path[-1] == discriminator_field:
                return  # This is a discriminator field, it's allowed to be a primitive

        # Otherwise, this is an error
        path_str = ".".join(path) if path else "(root)"
        raise ValueError(
            f'All leaf values in sweep parameters must be {{"values": [...]}}, '
            f"but found {type(obj).__name__} at path '{path_str}': {obj}"
        )


def generate_grid_combinations(parameters: dict[str, Any]) -> list[dict[str, Any]]:
    """Generate all combinations for a grid search from parameter specifications.

    All leaf values (except discriminator fields) must be {"values": [...]}.
    Discriminated lists use discriminator values in flattened keys instead of indices.

    Args:
        parameters: Nested dict/list structure where all leaves are {"values": [...]}

    Returns:
        List of parameter combinations with flattened keys (e.g., "loss_metric_configs.Loss1.coeff")

    Example:
        >>> params = {
        ...     "seed": {"values": [0, 1]},
        ...     "loss_metric_configs": [
        ...         {
        ...             "classname": "ImportanceMinimalityLoss",
        ...             "coeff": {"values": [0.1, 0.2]},
        ...         }
        ...     ],
        ... }
        >>> combos = generate_grid_combinations(params)
        >>> len(combos)
        4
        >>> combos[0]["seed"]
        0
        >>> combos[0]["loss_metric_configs.ImportanceMinimalityLoss.coeff"]
        0.1
    """
    # Extract all value specs with their flattened paths
    value_specs: list[tuple[str, list[Any]]] = []
    _extract_value_specs_from_sweep_params(parameters, [], value_specs)

    # Validate all non-discriminator leaves have {"values": [...]}
    _validate_sweep_params_have_values(parameters, [])

    if not value_specs:
        # No value specs found, return single empty combination
        return [{}]

    # Generate cartesian product of all value specs
    keys, value_lists = zip(*value_specs, strict=True)
    all_value_combinations = list(itertools.product(*value_lists))

    # Create flattened dicts for each combination
    combinations: list[dict[str, Any]] = []
    for value_combo in all_value_combinations:
        combo_dict = dict(zip(keys, value_combo, strict=True))
        combinations.append(combo_dict)

    return combinations


RunType = Literal["spd", "train", "clustering/runs", "clustering/ensembles", "clustering/harvests"]

RUN_TYPE_ABBREVIATIONS: Final[dict[RunType, str]] = {
    "spd": "s",
    "train": "t",
    "clustering/runs": "c",
    "clustering/ensembles": "e",
    "clustering/harvests": "ch",
}


def generate_run_id(run_type: RunType) -> str:
    """Generate a unique run identifier.

    Format: `{type_abbr}-{random_hex}`
    """
    type_abbr = RUN_TYPE_ABBREVIATIONS[run_type]
    return f"{type_abbr}-{secrets.token_hex(4)}"


def parse_config(config_path: Path | str | None, config_json: str | None) -> Config:
    """Parse a Config from either a file path or a JSON string. Exactly one must be provided."""
    assert (config_path is not None) != (config_json is not None), (
        "Need exactly one of config_path and config_json"
    )
    if config_path is not None:
        return Config.from_file(config_path)
    assert config_json is not None
    return Config(**json.loads(config_json.removeprefix("json:")))


def parse_sweep_params(sweep_params_json: str | None) -> dict[str, Any] | None:
    """Parse sweep parameters from a JSON string, or return None if not provided."""
    if sweep_params_json is None:
        return None
    return json.loads(sweep_params_json.removeprefix("json:"))


class ExecutionStamp(NamedTuple):
    run_id: str
    snapshot_branch: str
    commit_hash: str
    run_type: RunType

    @classmethod
    def create(
        cls,
        run_type: RunType,
        create_snapshot: bool,
    ) -> "ExecutionStamp":
        """Create an execution stamp, possibly including a git snapshot branch."""
        run_id = generate_run_id(run_type)
        snapshot_branch: str
        commit_hash: str

        if create_snapshot:
            snapshot_branch, commit_hash = create_git_snapshot(snapshot_id=run_id)
            logger.info(f"Created git snapshot branch: {snapshot_branch} ({commit_hash[:8]})")
        else:
            snapshot_branch = repo_current_branch()
            if repo_is_clean():
                commit_hash = repo_current_commit_hash()
                logger.info(f"Using current branch: {snapshot_branch} ({commit_hash[:8]})")
            else:
                commit_hash = "none"
                logger.info(
                    f"Using current branch: {snapshot_branch} (uncommitted changes, no commit hash)"
                )

        return ExecutionStamp(
            run_id=run_id,
            snapshot_branch=snapshot_branch,
            commit_hash=commit_hash,
            run_type=run_type,
        )

    @property
    def out_dir(self) -> Path:
        """Get the output directory for this execution stamp."""
        run_dir = SPD_OUT_DIR / self.run_type / self.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir


_NO_ARG_PARSSED_SENTINEL = object()


def read_noneable_str(value: str) -> str | None:
    """Read a string that may be 'None' and convert to None."""
    if value == "None":
        return None
    return value


def run_locally(
    commands: list[str],
    parallel: bool = False,
    track_resources: bool = False,
) -> dict[str, dict[str, float]] | None:
    """Run commands locally instead of via SLURM.

    Useful for testing and for --local mode in clustering pipeline.

    Args:
        commands: List of shell commands to run
        parallel: If True, run all commands in parallel. If False, run sequentially.
        track_resources: If True, track and return resource usage via /usr/bin/time

    Returns:
        If track_resources is True, dict mapping commands to resource metrics.
        Metrics include: K (avg memory KB), M (max memory KB), P (CPU %),
        S (system CPU sec), U (user CPU sec), e (wall time sec).
        Otherwise None.
    """
    n_commands = len(commands)
    resources: dict[str, dict[str, float]] = {}
    resource_files: list[Path] = []

    # Wrap commands with /usr/bin/time if resource tracking is requested
    if track_resources:
        wrapped_commands: list[str] = []
        for cmd in commands:
            # Create a unique temp file for resource tracking output
            fd, resource_file_path = tempfile.mkstemp(suffix=".resources")
            os.close(fd)  # Close fd, we just need the path for /usr/bin/time -o
            resource_file = Path(resource_file_path)
            resource_files.append(resource_file)
            # Use /usr/bin/time to track comprehensive resource usage
            # K=avg total mem, M=max resident, P=CPU%, S=system time, U=user time, e=wall time
            wrapped_cmd = (
                f'/usr/bin/time -f "K:%K M:%M P:%P S:%S U:%U e:%e" -o {resource_file} {cmd}'
            )
            wrapped_commands.append(wrapped_cmd)
        commands_to_run = wrapped_commands
    else:
        commands_to_run = commands

    try:
        if not parallel:
            logger.section(f"LOCAL EXECUTION: Running {n_commands} tasks serially")
            for i, cmd in enumerate(commands_to_run, 1):
                logger.info(f"[{i}/{n_commands}] Running: {commands[i - 1]}")
                subprocess.run(cmd, shell=True, check=True)
            logger.section("LOCAL EXECUTION COMPLETE")
        else:
            logger.section(f"LOCAL EXECUTION: Starting {n_commands} tasks in parallel")
            procs: list[subprocess.Popen[bytes]] = []

            for i, cmd in enumerate(commands_to_run, 1):
                logger.info(f"[{i}/{n_commands}] Starting: {commands[i - 1]}")
                proc = subprocess.Popen(cmd, shell=True)
                procs.append(proc)

            logger.section("WAITING FOR ALL TASKS TO COMPLETE")
            for proc, cmd in zip(procs, commands, strict=True):  # noqa: B007
                proc.wait()
                if proc.returncode != 0:
                    logger.error(f"Process {proc.pid} failed with exit code {proc.returncode}")
            logger.section("LOCAL EXECUTION COMPLETE")

        # Read resource usage results
        if track_resources:
            for cmd, resource_file in zip(commands, resource_files, strict=True):
                if resource_file.exists():
                    # Parse format: "K:123 M:456 P:78% S:1.23 U:4.56 e:7.89"
                    output = resource_file.read_text().strip()
                    metrics: dict[str, float] = {}

                    for part in output.split():
                        if ":" in part:
                            key, value = part.split(":", 1)
                            # Remove % sign from CPU percentage
                            value = value.rstrip("%")
                            try:
                                metrics[key] = float(value)
                            except ValueError:
                                logger.warning(f"Could not parse {key}:{value} for command: {cmd}")

                    resources[cmd] = metrics
                else:
                    logger.warning(f"Resource file not found for: {cmd}")

            # Log comprehensive resource usage table
            logger.section("RESOURCE USAGE RESULTS")
            for cmd, metrics in resources.items():
                logger.info(f"Command: {cmd}")
                logger.info(
                    f"  Time: {metrics.get('e', 0):.2f}s wall, "
                    f"{metrics.get('U', 0):.2f}s user, "
                    f"{metrics.get('S', 0):.2f}s system"
                )
                logger.info(
                    f"  Memory: {metrics.get('M', 0) / 1024:.1f} MB peak, "
                    f"{metrics.get('K', 0) / 1024:.1f} MB avg"
                )
                logger.info(f"  CPU: {metrics.get('P', 0):.1f}%")

    finally:
        # Clean up temp files
        if track_resources:
            for resource_file in resource_files:
                if resource_file.exists():
                    resource_file.unlink()

    return resources if track_resources else None
