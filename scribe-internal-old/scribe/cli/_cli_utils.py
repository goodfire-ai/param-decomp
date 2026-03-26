"""Scribe CLI util functions."""

import subprocess
import os
import sys
from typing import Dict, Any


def get_python_path() -> str:
    """Get the Python interpreter path that the user would get from "which python"."""
    try:
        result = subprocess.run(
            ["which", "python"], capture_output=True, text=True, check=True
        )
        python_path = result.stdout.strip()
        os.environ["SCRIBE_PYTHON"] = python_path
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Fallback to sys.executable if "which python" fails
        os.environ["SCRIBE_PYTHON"] = sys.executable

    return os.environ["SCRIBE_PYTHON"]


def merge_settings_intelligently(
    new_settings: Dict[str, Any], existing_settings: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Merge new settings with existing settings intelligently.

    Args:
        new_settings: The new settings to apply
        existing_settings: The existing settings to merge with

    Returns:
        Merged settings dictionary

    For mcpServers, merges at the server level (preserves existing servers, adds/updates new ones).
    For other keys, new settings override existing ones.
    Existing settings not in new_settings are preserved.
    """
    merged = existing_settings.copy()

    for key, value in new_settings.items():
        if key == "mcpServers" and key in merged:
            # For mcpServers, merge at the server level
            merged[key].update(value)
        else:
            # For other keys, just override
            merged[key] = value

    return merged
