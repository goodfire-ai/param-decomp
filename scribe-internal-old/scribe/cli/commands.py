"""Commands for scribe CLI."""

import json
import os
import subprocess
import sys
import tempfile
import uuid
import click
from rich.console import Console
from scribe.cli.constants import DEFAULT_PROVIDER
from pathlib import Path

from scribe.providers._provider_utils import get_provider
from scribe.providers.claude import CLAUDE_COPILOT_SETTINGS
from scribe.providers.gemini import get_copilot_settings as get_gemini_copilot_settings
from scribe.providers.codex import get_copilot_settings as get_codex_copilot_settings
from scribe.cli._cli_utils import get_python_path, merge_settings_intelligently
from scribe.cli.export_transcript import export_transcript


console = Console()


def copilot_impl(args=None, provider_name: str = DEFAULT_PROVIDER, verbose=False):
    """Implementation of copilot command - supports multiple AI providers."""
    provider = get_provider(provider_name)
    provider_display = provider.get_provider_display_name()

    # Give terminal a nice title for Gemini CLI
    if provider_name == "gemini":
        os.environ["CLI_TITLE"] = "Scribe - Copilot"

    # Generate session ID
    session_id = str(uuid.uuid4())
    os.environ["SCRIBE_SESSION_ID"] = session_id

    try:
        click.echo(f"\n🚀 Launching {provider_display} in copilot mode...\n")
        python_path = get_python_path()

        if provider_name == "claude":
            # Workaround for Claude Code ENXIO bug when using --mcp-config/--settings
            # in VS Code/Cursor integrated terminals on Linux. Must be set BEFORE
            # creating temp files so they're created in the safe directory.
            # See: https://github.com/anthropics/claude-code/issues/15112
            claude_tmpdir = "/tmp/claude-tmp"
            os.makedirs(claude_tmpdir, exist_ok=True)
            os.environ["TMPDIR"] = claude_tmpdir

            # Create temporary file for settings and mcp config
            settings_file = tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            )
            json.dump(CLAUDE_COPILOT_SETTINGS, settings_file)
            settings_file.close()

            mcp_config_file = tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            )
            json.dump(provider.get_copilot_mcp_config(python_path), mcp_config_file)
            mcp_config_file.close()

            cmd = [
                "claude",
                "--mcp-config",
                mcp_config_file.name,
                "--settings",
                settings_file.name,
                "--append-system-prompt",
                "",
            ]

            # Add any additional args passed to scribe copilot
            if args:
                cmd.extend(args)

            subprocess.run(cmd)
        elif provider_name == "gemini":
            # Get the settings from the provider
            settings_content = get_gemini_copilot_settings(python_path)

            # Create .gemini directory if it doesn't exist
            gemini_dir = Path(".gemini")
            gemini_dir.mkdir(exist_ok=True)

            settings_file = gemini_dir / "settings.json"

            # If settings file exists, merge intelligently
            if settings_file.exists():
                try:
                    with open(settings_file, "r") as f:
                        existing_settings = json.load(f)

                    settings_content = merge_settings_intelligently(
                        settings_content, existing_settings
                    )
                except (json.JSONDecodeError, IOError):
                    # If we can't read existing settings, just use new ones
                    pass

            # Write the settings file
            with open(settings_file, "w") as f:
                json.dump(settings_content, f, indent=2)

            # Build and run the gemini command using provider base (supports npx fallback)
            cmd = provider.get_command_base()

            # Add any additional args passed to scribe copilot
            if args:
                cmd.extend(args)

            subprocess.run(cmd)
        elif provider_name == "codex":
            # Check if we have piped input (non-interactive mode)
            import select
            has_input = select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])
            
            if has_input:
                # Non-interactive mode: read stdin and use codex exec
                input_text = sys.stdin.read().strip()
                if input_text:
                    cmd = provider.get_command_base()
                    cmd.extend(["exec", input_text])
                    
                    # Add any additional args passed to scribe copilot
                    if args:
                        cmd.extend(args)
                    
                    subprocess.run(cmd)
                    return
            
            # Interactive mode: Build the codex command with -c overrides for MCP
            settings_content = get_codex_copilot_settings(python_path)

            # Extract scribe MCP config
            scribe_cfg = settings_content.get("mcp_servers", {}).get("scribe", {})
            mcp_command = scribe_cfg.get("command", python_path)
            mcp_args = scribe_cfg.get("args", [])
            mcp_env = scribe_cfg.get("env", {})

            cmd = provider.get_command_base()

            import json as _json

            # Command and args
            cmd.extend(["-c", f"mcp_servers.scribe.command={mcp_command}"])
            cmd.extend(["-c", f"mcp_servers.scribe.args={_json.dumps(mcp_args)}"])

            # Optional env passthroughs (like NOTEBOOK_OUTPUT_DIR)
            for _k, _v in mcp_env.items():
                cmd.extend(["-c", f"mcp_servers.scribe.env.{_k}={_v}"])

            # Add any additional args passed to scribe copilot
            if args:
                cmd.extend(args)

            subprocess.run(cmd)

        else:
            raise ValueError(f"Unsupported provider: {provider_name}")

    except FileNotFoundError:
        click.echo(f"Error: {provider_display} command not found.")
        click.echo(
            f"Please ensure {provider.get_provider_name()} CLI is installed and in PATH."
        )
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error launching {provider_display}: {e}")
        sys.exit(1)
    # finally:
    #     # Clean up temporary files
    #     if settings_file and os.path.exists(settings_file.name):
    #         os.remove(settings_file.name)


def export_transcript_impl():
    """Implementation of export transcript command."""
    return export_transcript()
