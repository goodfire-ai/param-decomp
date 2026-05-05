"""Helpers for stable component-subset selection."""

from pathlib import Path

SELECTED_COMPONENT_KEYS_FILENAME = "component_keys.txt"


def load_component_keys_file(path: str | Path) -> list[str]:
    """Load newline-delimited component keys, ignoring blanks and comments."""
    lines = Path(path).read_text().splitlines()
    return [line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#")]


def save_component_keys_file(path: str | Path, component_keys: list[str]) -> None:
    out = "\n".join(component_keys)
    Path(path).write_text(f"{out}\n" if out else "")


def get_subrun_component_keys_path(subrun_dir: Path) -> Path:
    return subrun_dir / SELECTED_COMPONENT_KEYS_FILENAME
