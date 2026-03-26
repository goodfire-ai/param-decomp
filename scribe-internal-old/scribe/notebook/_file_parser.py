"""Parse .py files with # %% cell markers into structured cells."""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ParsedCell:
    tag: str
    cell_type: Literal["code", "markdown"]
    source: str


def parse_named_cells(content: str) -> list[ParsedCell]:
    """Parse a .py file into named cells delimited by `# %% tag` markers.

    Each `# %% tag-name` line starts a new code cell.
    `# %% md:tag-name` starts a markdown cell — body lines have `# ` prefix stripped.
    Content before the first marker is ignored.
    Tags must be unique within a file.
    """
    lines = content.split("\n")
    cells: list[tuple[str, str, list[str]]] = []
    current_tag: str | None = None
    current_type: str = "code"
    current_lines: list[str] = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("# %%"):
            if current_tag is not None:
                cells.append((current_tag, current_type, current_lines))
            raw_tag = stripped[4:].strip()
            assert raw_tag, f"Cell marker must have a tag: `# %% my-tag`, got: `{stripped}`"
            if raw_tag.startswith("md:"):
                current_tag = raw_tag[3:]
                current_type = "markdown"
            else:
                current_tag = raw_tag
                current_type = "code"
            current_lines = []
        elif current_tag is not None:
            current_lines.append(line)

    if current_tag is not None:
        cells.append((current_tag, current_type, current_lines))

    seen: set[str] = set()
    dupes = {tag for tag, _, _ in cells if tag in seen or seen.add(tag)}  # type: ignore[func-returns-value]
    assert not dupes, f"Duplicate cell tags: {dupes}"

    result = []
    for tag, cell_type, raw_lines in cells:
        source = "\n".join(raw_lines).strip()
        if cell_type == "markdown":
            md_lines = []
            for ln in source.split("\n"):
                if ln.startswith("# "):
                    md_lines.append(ln[2:])
                elif ln == "#":
                    md_lines.append("")
                else:
                    md_lines.append(ln)
            source = "\n".join(md_lines)
        result.append(ParsedCell(tag=tag, cell_type=cell_type, source=source))
    return result
