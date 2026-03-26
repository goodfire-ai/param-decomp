"""
Codex CLI provider implementation.

Expected manual invocation example (current workflow):
codex -c mcp_servers.scribe.command="$(which python)" \
      -c mcp_servers.scribe.args='["-m", "scribe.notebook.notebook_mcp_server"]'

This provider encapsulates the above so scribe can launch Codex similarly to Gemini.
"""

import os
import subprocess
from typing import List

from .base import AICLIProvider


def get_copilot_settings(python_path: str) -> dict:
    """Return settings payload equivalent to the manual Codex -c flags.

    While Codex is typically configured via repeated -c flags, we keep a
    structured representation here to reuse the same merging path that Gemini
    uses in the CLI layer.
    """
    return {
        "mcp_servers": {
            "scribe": {
                "command": python_path,
                "args": ["-m", "scribe.notebook.notebook_mcp_server"],
                "env": {
                    # Location of notebook outputs - only include if set to avoid "null" string
                    **(
                        {}
                        if os.environ.get("NOTEBOOK_OUTPUT_DIR") is None
                        else {"NOTEBOOK_OUTPUT_DIR": os.environ.get("NOTEBOOK_OUTPUT_DIR")}
                    )
                },
            }
        }
    }


class CodexProvider(AICLIProvider):
    """Codex CLI provider."""

    def get_provider_name(self) -> str:
        return "codex"

    def get_provider_display_name(self) -> str:
        return "Codex CLI"

    def get_command_base(self) -> List[str]:
        # Prefer global codex, fallback to npx package
        try:
            subprocess.run(["which", "codex"], capture_output=True, check=True, timeout=3)
            return ["codex"]
        except (
            subprocess.CalledProcessError,
            FileNotFoundError,
            subprocess.TimeoutExpired,
        ):
            return ["npx", "@openai/codex"]

    def is_available(self) -> bool:
        # Try global codex first
        try:
            result = subprocess.run(
                ["which", "codex"], capture_output=True, text=True, timeout=5, check=False
            )
            if result.returncode == 0:
                return True
        except (
            subprocess.CalledProcessError,
            FileNotFoundError,
            subprocess.TimeoutExpired,
        ):
            pass

        # Try npx codex
        try:
            result = subprocess.run(
                ["npx", "@openai/codex", "--version"],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
            return result.returncode == 0
        except (
            subprocess.CalledProcessError,
            FileNotFoundError,
            subprocess.TimeoutExpired,
        ):
            return False


