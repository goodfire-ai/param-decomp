"""Format Jupyter outputs into compact text for LLM consumption."""

import re
from typing import Any

from fastmcp.utilities.types import Image

from scribe.notebook._notebook_server_utils import process_jupyter_outputs

MAX_OUTPUT_CHARS = 10_000


def format_outputs_compact(
    outputs: list[dict[str, Any]], images: list[Image]
) -> list[str | Image]:
    """Convert verbose Jupyter output JSON to minimal text + images."""
    if not outputs and not images:
        return ["(no output)"]

    text_parts: list[str] = []

    for output in outputs:
        match output.get("type"):
            case "text" if output.get("content"):
                text_parts.append(output["content"])
            case "result" if output.get("content") and output["content"] != "None":
                text_parts.append(f"=> {output['content']}")
            case "display" if output.get("content"):
                text_parts.append(output["content"])
            case "error":
                text_parts.append(_format_error(output))

    combined = "\n".join(text_parts) if text_parts else ""

    if len(combined) > MAX_OUTPUT_CHARS:
        combined = combined[:MAX_OUTPUT_CHARS] + f"\n\n[output truncated at {MAX_OUTPUT_CHARS:,} chars — see notebook for full output]"

    result: list[str | Image] = []
    if combined:
        result.append(combined)
    result.extend(images)
    if not combined and images:
        result.insert(0, f"[{len(images)} image(s) returned]")

    return result if result else ["(no output)"]


def _format_error(output: dict[str, Any]) -> str:
    error_name = output.get("name", "Error")
    error_msg = output.get("message", "")
    traceback = output.get("traceback", [])

    location = _extract_location(traceback)
    relevant_line = _extract_relevant_line(traceback, error_name)

    text = f"ERROR: {error_name}: {error_msg}"
    if location:
        text += f"\n  {location}"
    if relevant_line:
        text += f": {relevant_line}"
    return text


def _extract_location(traceback: list[str]) -> str:
    for line in traceback:
        if "line " not in line or ("Cell" not in line and "File" not in line):
            continue
        match = re.search(r"line (\d+)(?:.*in (.+))?", line)
        if match:
            loc = f"line {match.group(1)}"
            if match.group(2):
                loc += f", in {match.group(2)}"
            return loc
    return ""


def _extract_relevant_line(traceback: list[str], error_name: str) -> str:
    for line in reversed(traceback):
        line = line.strip()
        if not line or line.startswith("-") or line.startswith(error_name):
            continue
        if "line" in line.lower() or line.startswith("Cell") or line.startswith("File"):
            continue
        if line.startswith(">>>"):
            continue
        return line
    return ""


def has_error_output(raw_outputs: list[dict]) -> bool:
    return any(o.get("output_type") == "error" for o in raw_outputs)


def format_raw_outputs(
    raw_outputs: list[dict], session_id: str, save_images_locally: bool, compact: bool
) -> list[str | Image]:
    """Process raw Jupyter outputs into formatted text + images."""
    outputs, images = process_jupyter_outputs(
        raw_outputs, session_id=session_id, save_images_locally=save_images_locally,
    )
    if compact:
        return format_outputs_compact(outputs, images)
    result: list = [{"session_id": session_id, "outputs": outputs}]
    result.extend(images)
    return result
