"""Worker script that runs inside each SLURM job.

This script:
1. Reads the research question from the investigation metadata
2. Starts the app backend with an isolated database
3. Loads the SPD run and fetches model architecture info
4. Configures MCP server for Claude Code
5. Launches Claude Code with the investigation question
6. Handles cleanup on exit
"""

import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from types import FrameType
from typing import Any

import fire
import requests

from spd.investigate.agent_prompt import get_agent_prompt
from spd.investigate.schemas import InvestigationEvent
from spd.investigate.scripts.run_slurm import get_investigation_output_dir
from spd.log import logger


def write_mcp_config(inv_dir: Path, port: int) -> Path:
    """Write MCP configuration file for Claude Code."""
    mcp_config = {
        "mcpServers": {
            "spd": {
                "type": "http",
                "url": f"http://localhost:{port}/mcp",
            }
        }
    }
    config_path = inv_dir / "mcp_config.json"
    config_path.write_text(json.dumps(mcp_config, indent=2))
    return config_path


def write_claude_settings(inv_dir: Path) -> None:
    """Write Claude Code settings to pre-grant MCP tool permissions."""
    claude_dir = inv_dir / ".claude"
    claude_dir.mkdir(exist_ok=True)
    settings = {"permissions": {"allow": ["mcp__spd__*"]}}
    (claude_dir / "settings.json").write_text(json.dumps(settings, indent=2))


def find_available_port(start_port: int = 8000, max_attempts: int = 100) -> int:
    """Find an available port starting from start_port."""
    for offset in range(max_attempts):
        port = start_port + offset
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("localhost", port))
                return port
            except OSError:
                continue
    raise RuntimeError(
        f"Could not find available port in range {start_port}-{start_port + max_attempts}"
    )


def wait_for_backend(port: int, timeout: float = 120.0) -> bool:
    """Wait for the backend to become healthy."""
    url = f"http://localhost:{port}/api/health"
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                return True
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(1)
    return False


def load_run(port: int, wandb_path: str, context_length: int) -> None:
    """Load the SPD run into the backend. Raises on failure."""
    url = f"http://localhost:{port}/api/runs/load"
    params = {"wandb_path": wandb_path, "context_length": context_length}
    resp = requests.post(url, params=params, timeout=300)
    assert resp.status_code == 200, (
        f"Failed to load run {wandb_path}: {resp.status_code} {resp.text}"
    )


def fetch_model_info(port: int) -> dict[str, Any]:
    """Fetch model architecture info from the backend."""
    resp = requests.get(f"http://localhost:{port}/api/pretrain_info/loaded", timeout=30)
    assert resp.status_code == 200, f"Failed to fetch model info: {resp.status_code} {resp.text}"
    result: dict[str, Any] = resp.json()
    return result


def log_event(events_path: Path, event: InvestigationEvent) -> None:
    """Append an event to the events log."""
    with open(events_path, "a") as f:
        f.write(event.model_dump_json() + "\n")


def run_agent(
    wandb_path: str,
    inv_id: str,
    context_length: int = 128,
    max_turns: int = 50,
) -> None:
    """Run a single investigation agent.

    Args:
        wandb_path: WandB path of the SPD run.
        inv_id: Unique identifier for this investigation.
        context_length: Context length for prompts.
        max_turns: Maximum agentic turns before stopping (prevents runaway agents).
    """
    inv_dir = get_investigation_output_dir(inv_id)
    assert inv_dir.exists(), f"Investigation directory does not exist: {inv_dir}"

    # Read prompt from metadata
    metadata: dict[str, Any] = json.loads((inv_dir / "metadata.json").read_text())
    prompt = metadata["prompt"]

    write_claude_settings(inv_dir)

    events_path = inv_dir / "events.jsonl"
    (inv_dir / "explanations.jsonl").touch()

    log_event(
        events_path,
        InvestigationEvent(
            event_type="start",
            message=f"Investigation {inv_id} starting",
            details={"wandb_path": wandb_path, "inv_id": inv_id, "prompt": prompt},
        ),
    )

    port = find_available_port()
    logger.info(f"[{inv_id}] Using port {port}")

    log_event(
        events_path,
        InvestigationEvent(
            event_type="progress",
            message=f"Starting backend on port {port}",
            details={"port": port},
        ),
    )

    # Start backend with investigation configuration
    env = os.environ.copy()
    env["SPD_INVESTIGATION_DIR"] = str(inv_dir)

    backend_cmd = [
        sys.executable,
        "-m",
        "spd.app.backend.server",
        "--port",
        str(port),
    ]

    backend_log_path = inv_dir / "backend.log"
    backend_log = open(backend_log_path, "w")  # noqa: SIM115 - managed manually
    backend_proc = subprocess.Popen(
        backend_cmd,
        env=env,
        stdout=backend_log,
        stderr=subprocess.STDOUT,
    )

    def cleanup(signum: int | None = None, frame: FrameType | None = None) -> None:
        _ = frame
        logger.info(f"[{inv_id}] Cleaning up...")
        if backend_proc.poll() is None:
            backend_proc.terminate()
            try:
                backend_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                backend_proc.kill()
        backend_log.close()
        if signum is not None:
            sys.exit(1)

    signal.signal(signal.SIGTERM, cleanup)
    signal.signal(signal.SIGINT, cleanup)

    try:
        logger.info(f"[{inv_id}] Waiting for backend...")
        if not wait_for_backend(port):
            log_event(
                events_path,
                InvestigationEvent(event_type="error", message="Backend failed to start"),
            )
            raise RuntimeError("Backend failed to start")

        logger.info(f"[{inv_id}] Backend ready, loading run...")
        log_event(
            events_path,
            InvestigationEvent(event_type="progress", message="Backend ready, loading run"),
        )

        load_run(port, wandb_path, context_length)

        logger.info(f"[{inv_id}] Run loaded, fetching model info...")
        model_info = fetch_model_info(port)

        logger.info(f"[{inv_id}] Launching Claude Code...")
        log_event(
            events_path,
            InvestigationEvent(
                event_type="progress", message="Run loaded, launching Claude Code agent"
            ),
        )

        agent_prompt = get_agent_prompt(
            wandb_path=wandb_path,
            prompt=prompt,
            model_info=model_info,
        )

        (inv_dir / "agent_prompt.md").write_text(agent_prompt)

        mcp_config_path = write_mcp_config(inv_dir, port)
        logger.info(f"[{inv_id}] MCP config written to {mcp_config_path}")

        claude_output_path = inv_dir / "claude_output.jsonl"
        claude_cmd = [
            "claude",
            "--print",
            "--verbose",
            "--output-format",
            "stream-json",
            "--max-turns",
            str(max_turns),
            # MCP: only our backend, no inherited servers
            "--mcp-config",
            str(mcp_config_path),
            # Permissions: only MCP tools, deny everything else
            "--permission-mode",
            "dontAsk",
            "--allowedTools",
            "mcp__spd__*",
            # Isolation: skip all user/project settings (no plugins, no inherited config)
            "--setting-sources",
            "",
            "--model",
            "opus",
        ]

        logger.info(f"[{inv_id}] Starting Claude Code (max_turns={max_turns})...")
        logger.info(f"[{inv_id}] Monitor with: tail -f {claude_output_path}")

        with open(claude_output_path, "w") as output_file:
            claude_proc = subprocess.Popen(
                claude_cmd,
                stdin=subprocess.PIPE,
                stdout=output_file,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=str(inv_dir),
            )

            assert claude_proc.stdin is not None
            claude_proc.stdin.write(agent_prompt)
            claude_proc.stdin.close()

            claude_proc.wait()

        log_event(
            events_path,
            InvestigationEvent(
                event_type="complete",
                message="Investigation complete",
                details={"exit_code": claude_proc.returncode},
            ),
        )

        logger.info(f"[{inv_id}] Investigation complete")

    except Exception as e:
        log_event(
            events_path,
            InvestigationEvent(
                event_type="error",
                message=f"Agent failed: {e}",
                details={"error_type": type(e).__name__},
            ),
        )
        logger.error(f"[{inv_id}] Failed: {e}")
        raise
    finally:
        cleanup()


def cli() -> None:
    fire.Fire(run_agent)


if __name__ == "__main__":
    cli()
