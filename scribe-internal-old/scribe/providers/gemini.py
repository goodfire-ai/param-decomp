"""
Gemini CLI provider implementation.

Docs on configuring Gemini: https://github.com/google-gemini/gemini-cli/blob/main/docs/cli/configuration.md
"""

import subprocess
import os
from typing import List

from .base import AICLIProvider


def get_copilot_settings(python_path: str) -> dict:
    # Gemini-specific settings
    # Note: Unlike Claude, Gemini doesn't use permissions/tools allow lists
    # MCP servers are configured directly in mcpServers section
    config = {
        # Empty by default - mcpServers will be added dynamically
        "hideBanner": True,
        "theme": "Ayu",
        "usageStatisticsEnabled": False,
        "autoAccept": True,
        "mcpServers": {
            "scribe": {
                "command": python_path,
                "args": ["-m", "scribe.notebook.notebook_mcp_server"],
                "env": {
                    "SCRIBE_PROVIDER": "gemini",
                    # Save notebook images to tmp directory for agent to view them if the agent doesn't support images from MCP tools
                    # e.g. Gemini
                    "SCRIBE_SAVE_NB_IMAGES": True,
                    # Location of notebook outputs - only include if set to avoid "null" string
                    **(
                        {}
                        if os.environ.get("NOTEBOOK_OUTPUT_DIR") is None
                        else {
                            "NOTEBOOK_OUTPUT_DIR": os.environ.get("NOTEBOOK_OUTPUT_DIR")
                        }
                    ),
                },
                "trust": True,
            }
        },
    }

    return config


class GeminiProvider(AICLIProvider):
    """Gemini CLI provider."""

    def get_provider_name(self) -> str:
        return "gemini"

    def get_provider_display_name(self) -> str:
        return "Gemini CLI"

    def get_command_base(self) -> List[str]:
        # Check if global gemini is available, fallback to npx
        try:
            subprocess.run(
                ["which", "gemini"], capture_output=True, check=True, timeout=3
            )
            return ["gemini"]
        except (
            subprocess.CalledProcessError,
            FileNotFoundError,
            subprocess.TimeoutExpired,
        ):
            return ["npx", "@google/gemini-cli"]

    # def get_settings_dir(self) -> str:
    #     return ".gemini"

    # def get_settings_filename(self) -> str:
    #     return "settings.json"

    # def build_copilot_command(self, args: List[str]) -> List[str]:
    #     base_cmd = self.get_command_base()
    #     # Add MCP server names if configured
    #     cmd = base_cmd[:]

    #     # Check if we have MCP servers configured
    #     settings_path = Path(self.get_settings_dir()) / self.get_settings_filename()
    #     if settings_path.exists():
    #         try:
    #             with open(settings_path) as f:
    #                 config = json.load(f)
    #                 if "mcpServers" in config:
    #                     # Add allowed MCP server names
    #                     for server_name in config["mcpServers"].keys():
    #                         cmd.extend(["--allowed-mcp-server-names", server_name])
    #         except (json.JSONDecodeError, IOError):
    #             pass

    # return cmd + args

    # def build_agent_command(self, prompt: str, args: List[str]) -> List[str]:
    #     base_cmd = self.get_command_base()
    #     cmd = base_cmd[:]

    #     # Check if we have MCP servers configured
    #     settings_path = Path(self.get_settings_dir()) / self.get_settings_filename()
    #     if settings_path.exists():
    #         try:
    #             with open(settings_path) as f:
    #                 config = json.load(f)
    #                 if "mcpServers" in config:
    #                     # Add allowed MCP server names
    #                     for server_name in config["mcpServers"].keys():
    #                         cmd.extend(["--allowed-mcp-server-names", server_name])
    #         except (json.JSONDecodeError, IOError):
    #             pass

    #     return cmd + ["--prompt-interactive", prompt] + args

    # def build_chat_command(self, session_id: str, prompt: Optional[str] = None) -> List[str]:
    #     base_cmd = self.get_command_base()
    #     cmd = base_cmd[:]

    #     # Check if we have MCP servers configured
    #     settings_path = Path(self.get_settings_dir()) / self.get_settings_filename()
    #     if settings_path.exists():
    #         try:
    #             with open(settings_path) as f:
    #                 config = json.load(f)
    #                 if "mcpServers" in config:
    #                     # Add allowed MCP server names
    #                     for server_name in config["mcpServers"].keys():
    #                         cmd.extend(["--allowed-mcp-server-names", server_name])
    #         except (json.JSONDecodeError, IOError):
    #             pass

    #     # Add session resumption - adjust based on actual Gemini CLI interface
    #     cmd.extend(["--session", session_id])

    #     if prompt:
    #         cmd.extend(["--prompt", prompt])

    #     return cmd

    def is_available(self) -> bool:
        # Try global gemini first
        try:
            result = subprocess.run(
                ["which", "gemini"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if result.returncode == 0:
                return True
        except (
            subprocess.CalledProcessError,
            FileNotFoundError,
            subprocess.TimeoutExpired,
        ):
            pass

        # Try npx version
        try:
            result = subprocess.run(
                ["npx", "@google/gemini-cli", "--version"],
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

    # def setup_settings(self, mcp_config: Optional[Dict[str, Any]] = None) -> None:
    #     """Create or update Gemini CLI settings (simple overwrite, no merge)."""

    #     # Get settings content
    #     settings_content = self.get_settings_content()

    #     # If dynamic MCP config provided, merge it
    #     if mcp_config and "mcpServers" in mcp_config:
    #         settings_content["mcpServers"] = mcp_config["mcpServers"]

    #     # Write settings to file
    #     settings_dir = Path(self.get_settings_dir())
    #     settings_path = settings_dir / self.get_settings_filename()
    #     settings_dir.mkdir(exist_ok=True)

    #     try:
    #         with open(settings_path, "w") as f:
    #             json.dump(settings_content, f, indent=2)
    #     except IOError as e:
    #         raise RuntimeError(f"Failed to write {self.get_provider_name()} settings: {e}")
