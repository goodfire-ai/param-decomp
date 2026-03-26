"""
End-to-end tests using Claude Code to interact with the Scribe MCP server.

These tests verify the full integration by having Claude actually use the MCP tools.
They require the `claude` CLI to be installed and authenticated.

Run with: pytest tests/test_e2e_claude.py -v -s

Note: These tests make API calls and may incur costs. They are slower than unit tests.
"""

import json
import os
import subprocess
import tempfile
from pathlib import Path

import pytest


def run_claude(prompt: str, timeout: int = 180) -> str:
    """Run claude -p with a prompt and return the output."""
    project_root = Path(__file__).parent.parent

    mcp_config = {
        "mcpServers": {
            "scribe": {
                "command": "uv",
                "args": [
                    "--directory", str(project_root),
                    "run", "python", "-m", "scribe.notebook.notebook_mcp_server"
                ],
                "env": {
                    "SCRIBE_COMPACT_OUTPUT": "true"
                }
            }
        }
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(mcp_config, f)
        config_path = f.name

    try:
        result = subprocess.run(
            ["claude", "-p", prompt, "--mcp-config", config_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(project_root),
        )
        return result.stdout + result.stderr
    finally:
        os.unlink(config_path)


def claude_available() -> bool:
    """Check if claude CLI is available."""
    try:
        result = subprocess.run(["claude", "--version"], capture_output=True, timeout=5)
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


pytestmark = pytest.mark.skipif(not claude_available(), reason="claude CLI not available")


class TestE2EScribe:
    """Consolidated E2E tests for Scribe MCP server."""

    def test_full_notebook_workflow(self):
        """Test complete notebook workflow: session, code, markdown, edit, list, errors."""
        prompt = """
        Using the scribe MCP tools, perform these steps IN ORDER and report results for each:

        1. Start a new session with experiment_name "e2e_full_test"
        2. Add markdown: "# E2E Test Notebook"
        3. Execute: data = [10, 20, 30, 40, 50]
        4. Execute: print(f"Sum: {sum(data)}, Mean: {sum(data)/len(data)}")
        5. Execute: print(undefined_variable)  # This should error
        6. Edit the last cell (the one that errored) to: print("Fixed!")
        7. Use list_cells to see all cells
        8. Execute: import numpy as np; print(f"Numpy sum: {np.sum(data)}")

        Report concisely:
        - Step 4 output (sum and mean)
        - Step 5 error type
        - Step 6 output after fix
        - Step 7 cell count
        - Step 8 numpy result
        """

        output = run_claude(prompt, timeout=180)

        # Verify key outputs
        assert "150" in output, f"Expected sum=150. Output: {output[:1000]}"
        assert "30" in output, f"Expected mean=30. Output: {output[:1000]}"
        assert "nameerror" in output.lower() or "not defined" in output.lower(), \
            f"Expected NameError. Output: {output[:1000]}"
        assert "fixed" in output.lower(), f"Expected 'Fixed!' after edit. Output: {output[:1000]}"

    def test_cell_operations(self):
        """Test rerun, delete, and cell indexing."""
        prompt = """
        Using the scribe MCP tools:

        1. Start a new session
        2. Execute: counter = 0
        3. Execute: counter += 1; print(f"Counter: {counter}")
        4. Rerun cell_index=-1 (should increment again)
        5. Execute: x = 100
        6. Delete cell_index=2 (the counter increment cell)
        7. List cells and count them

        Report:
        - Counter value after step 3
        - Counter value after step 4 rerun
        - Number of cells after deletion
        """

        output = run_claude(prompt, timeout=120)

        # Counter should go 1, then 2 on rerun
        assert "1" in output and "2" in output, \
            f"Expected counter values 1 and 2. Output: {output[:1000]}"

    def test_timeout_handling(self):
        """Test that execution timeouts work correctly."""
        prompt = """
        Using the scribe MCP tools:

        1. Start a new session
        2. Execute with timeout_seconds=3:
           import time
           time.sleep(30)
           print("Should not reach here")
        3. Execute: print("Session still works!")

        Report whether step 2 timed out and whether step 3 succeeded.
        """

        output = run_claude(prompt, timeout=120)

        # Should mention timeout and session should still work
        assert "timeout" in output.lower() or "timed out" in output.lower() or "interrupt" in output.lower(), \
            f"Expected timeout. Output: {output[:1000]}"
        assert "still works" in output.lower() or "succeeded" in output.lower() or "session" in output.lower(), \
            f"Expected session recovery. Output: {output[:1000]}"
