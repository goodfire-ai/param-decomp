"""Claude CLI provider implementation."""

import os
import subprocess

from scribe.providers.base import AICLIProvider

# Claude-specific settings
CLAUDE_COPILOT_SETTINGS = {
    "permissions": {
        "allow": [
            "mcp__scribe__start_new_session",
            "mcp__scribe__restart_kernel",
            "mcp__scribe__shutdown_session",
            "mcp__scribe__run_cell",
            "mcp__scribe__run_cells",
            "mcp__scribe__start_cell",
            "mcp__scribe__start_cells",
            "mcp__scribe__get_output",
            "mcp__scribe__cancel",
        ]
    },
    "enableAllProjectMcpServers": True,
    "enabledMcpjsonServers": ["scribe"],
}


class ClaudeProvider(AICLIProvider):
    """Claude CLI provider - preserves exact existing behavior."""

    def get_provider_name(self) -> str:
        return "claude"

    def get_copilot_mcp_config(self, python_path: str) -> dict:
        config = {
            "mcpServers": {
                "scribe": {
                    "command": python_path,
                    "args": ["-m", "scribe.notebook.notebook_mcp_server"],
                    "env": {
                        # Location of notebook outputs - only include if set to avoid "null" string
                        **(
                            {}
                            if os.environ.get("NOTEBOOK_OUTPUT_DIR") is None
                            else {
                                "NOTEBOOK_OUTPUT_DIR": os.environ.get(
                                    "NOTEBOOK_OUTPUT_DIR"
                                )
                            }
                        )
                    },
                }
            }
        }

        return config

    def get_provider_display_name(self) -> str:
        return "Claude Code CLI"

    # def get_command_base(self) -> List[str]:
    #     return ["claude"]

    # def get_settings_dir(self) -> str:
    #     return ".claude"

    # def get_settings_filename(self) -> str:
    #     return "settings.json"

    # def get_settings_content(self) -> Dict[str, Any]:
    #     return CLAUDE_SETTINGS.copy()

    # def build_copilot_command(self, args: List[str]) -> List[str]:
    #     return ["claude"] + args

    # def build_agent_command(self, prompt: str, args: List[str]) -> List[str]:
    #     return ["claude", prompt] + args

    # def build_chat_command(self, session_id: str, prompt: Optional[str] = None) -> List[str]:
    #     if prompt:
    #         return ["claude", prompt, "-r", session_id]
    #     return ["claude", "-r", session_id]

    def is_available(self) -> bool:
        try:
            result = subprocess.run(
                ["which", "claude"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            return result.returncode == 0
        except (
            subprocess.CalledProcessError,
            FileNotFoundError,
            subprocess.TimeoutExpired,
        ):
            return False

    # def get_instructions_template_name(self) -> str:
    #     return "_claude.template.md"

    # def get_instructions_file_name(self) -> str:
    #     return "CLAUDE.md"

    # def get_settings_template_name(self) -> str:
    #     return "_settings.template.json"

    # def supports_mcp_config_flag(self) -> bool:
    #     return True

    # def setup_settings(self, mcp_config: Optional[Dict[str, Any]] = None) -> None:
    #     """Create or update Claude settings with merge behavior for existing settings."""

    #     # Get settings content
    #     settings_dir = Path(self.get_settings_dir())
    #     settings_path = settings_dir / self.get_settings_filename()
    #     settings_dir.mkdir(exist_ok=True)

    #     settings_content = self.get_settings_content()

    #     # For Claude, preserve existing merge behavior
    #     if settings_path.exists():
    #         try:
    #             with open(settings_path) as f:
    #                 existing_settings = json.load(f)
    #             # Remove old hooks if they exist
    #             existing_settings.pop("hooks", None)
    #             # Merge with our settings
    #             existing_settings.update(settings_content)
    #             settings_content = existing_settings
    #         except (json.JSONDecodeError, IOError):
    #             # If we can't read existing settings, just use new settings
    #             pass

    #     try:
    #         with open(settings_path, "w") as f:
    #             json.dump(settings_content, f, indent=2)
    #     except IOError as e:
    #         raise RuntimeError(f"Failed to write {self.get_provider_name()} settings: {e}")
