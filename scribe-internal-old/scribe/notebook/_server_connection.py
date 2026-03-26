"""Manage the Jupyter server subprocess and provide HTTP communication."""

import atexit
import os
import secrets
import signal
import subprocess
import sys
from typing import Any

import requests

from scribe.notebook._notebook_server_utils import (
    check_server_health,
    cleanup_scribe_server,
    find_safe_port,
    start_scribe_server,
)

# Server state
_process: subprocess.Popen | None = None
_port: int | None = None
_url: str | None = None
_token: str | None = None
_is_external: bool = False


def get_token() -> str:
    global _token
    if not _token and not _is_external:
        _token = secrets.token_urlsafe(32)
    return _token or ""


def get_url() -> str | None:
    return _url


def ensure_running() -> str:
    """Ensure a Jupyter server is running and return its URL."""
    global _process, _port, _url, _is_external

    if "SCRIBE_PORT" in os.environ:
        _port = int(os.environ["SCRIBE_PORT"])
        _url = f"http://127.0.0.1:{_port}"
        _is_external = True
        return _url

    if _process and _process.poll() is None:
        return _url

    _is_external = False
    port = find_safe_port()
    assert port, "Could not find an available port for Jupyter server"

    token = get_token()
    notebook_output_dir = os.environ.get("NOTEBOOK_OUTPUT_DIR")
    _process = start_scribe_server(port, token, notebook_output_dir)
    _port = port
    _url = f"http://127.0.0.1:{port}"

    atexit.register(cleanup)
    signal.signal(signal.SIGTERM, lambda sig, frame: cleanup())
    signal.signal(signal.SIGINT, lambda sig, frame: cleanup())

    print(f"Started managed Jupyter server at {_url}", file=sys.stderr)
    return _url


def cleanup():
    global _process, _token
    if _process and not _is_external:
        cleanup_scribe_server(_process)
        _process = None
        _token = None


def post(endpoint: str, body: dict) -> dict[str, Any]:
    """POST to a server endpoint. Starts the server if needed."""
    url = ensure_running()
    token = get_token()
    headers = {"Authorization": f"token {token}"} if token else {}
    response = requests.post(
        f"{url}/api/scribe/{endpoint}",
        json=body,
        headers=headers,
        timeout=(5, None),
    )
    response.raise_for_status()
    return response.json()


def get_status() -> dict[str, Any]:
    if not _url:
        return {"status": "not_started", "url": None, "port": None, "health": "unknown",
                "is_external": False, "will_shutdown_on_exit": True}

    health = "healthy" if (_port and check_server_health(_port)) else "unreachable"
    process_running = _is_external or (_process and _process.poll() is None)

    return {
        "status": "running" if process_running else "stopped",
        "url": _url, "port": _port,
        "vscode_url": f"{_url}/?token={get_token()}",
        "health": health,
        "is_external": _is_external,
        "will_shutdown_on_exit": not _is_external,
    }
