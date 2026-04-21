"""
Development server launcher for SPD app.

Starts backend and frontend with:
  - Automatic port detection (with --strictPort for Vite)
  - HTTP health checks that validate status codes (and optional content)
  - Fail-fast if a child dies during startup
  - Graceful shutdown (TERM -> KILL) of process groups
  - Clear logging & dependency checks
"""

from __future__ import annotations

import atexit
import contextlib
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from types import FrameType
from typing import TextIO

import requests


class AnsiEsc(StrEnum):
    GREEN = "\033[0;32m"
    YELLOW = "\033[1;33m"
    RED = "\033[0;31m"
    DIM = "\033[2m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    RESET = "\033[0m"


APP_DIR = Path(__file__).parent.resolve()
LOGS_DIR = APP_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
LOGFILE = LOGS_DIR / f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"

DEFAULT_STARTUP_TIMEOUT_SECONDS = 90
BACKEND_DEFAULT_START = 8000
FRONTEND_DEFAULT_START = 5173


def _require_bins(*bins: str) -> None:
    missing = [b for b in bins if shutil.which(b) is None]
    if missing:
        print(
            f"{AnsiEsc.RED}✗ Missing dependencies:{AnsiEsc.RESET} {', '.join(missing)}",
            file=sys.stderr,
        )
        sys.exit(1)


def is_port_in_use(port: int) -> bool:
    """Best-effort check: try binding on loopback IPv4 and IPv6."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s4:
        s4.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s4.bind(("127.0.0.1", port))
        except OSError:
            return True

    try:
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as s6:
            s6.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s6.bind(("::1", port))
            except OSError:
                return True
    except OSError:
        pass

    return False


def find_available_port(start_port: int) -> int:
    """Find an available port in [start_port, start_port+100)."""
    for port in range(start_port, start_port + 100):
        if not is_port_in_use(port):
            return port
    print(
        f"{AnsiEsc.RED}✗{AnsiEsc.RESET} Could not find available port after checking 100 ports from {start_port}",
        file=sys.stderr,
    )
    sys.exit(1)


def _tcp_open(host: str, port: int, timeout: float = 0.5) -> bool:
    """Returns True if a TCP connection can be established."""
    with contextlib.suppress(OSError), socket.create_connection((host, port), timeout=timeout):
        return True
    return False


def _spawn(
    cmd: list[str],
    cwd: Path,
    env: dict[str, str] | None,
    logfile: TextIO,
) -> subprocess.Popen[str]:
    """Spawn a process in its own process group, streaming stdout/stderr to logfile."""
    try:
        return subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=logfile,
            stderr=subprocess.STDOUT,
            bufsize=1,
            text=True,
            preexec_fn=os.setpgrp,
            env=env,
        )
    except FileNotFoundError as e:
        print(
            f"{AnsiEsc.RED}✗ Failed to start:{AnsiEsc.RESET} {' '.join(cmd)}\n"
            f"{AnsiEsc.DIM}{e}{AnsiEsc.RESET}",
            file=sys.stderr,
        )
        sys.exit(1)


@dataclass(frozen=True)
class HealthCheck:
    url: str
    ok_statuses: set[int]
    timeout: float = 1.0
    headers: dict[str, str] | None = None
    allow_redirects: bool = False
    body_predicate: Callable[[requests.Response], bool] | None = None


class AppRunner:
    """Manages backend and frontend processes with proper cleanup on signals."""

    def __init__(self, startup_timeout_seconds: int):
        self.backend_process: subprocess.Popen[str] | None = None
        self.frontend_process: subprocess.Popen[str] | None = None
        self.cleanup_called = False
        self.startup_timeout_seconds = startup_timeout_seconds

        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "spd-dev-launcher/1.0"})

    def _kill_process_group(self, proc: subprocess.Popen[str], sig: int) -> None:
        if proc.poll() is not None:
            return
        with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
            os.killpg(os.getpgid(proc.pid), sig)

    def cleanup(self) -> None:
        """Cleanup all running processes (process groups)."""
        if self.cleanup_called:
            return
        self.cleanup_called = True

        print("\nShutting down...", flush=True)

        procs = [p for p in (self.backend_process, self.frontend_process) if p]
        # Try graceful first
        for p in procs:
            self._kill_process_group(p, signal.SIGTERM)

        # Wait briefly
        deadline = time.time() + 2.0
        for p in procs:
            if not p:
                continue
            remaining = max(0.0, deadline - time.time())
            with contextlib.suppress(subprocess.TimeoutExpired):
                p.wait(timeout=remaining)

        # Force kill if still alive
        for p in procs:
            if p and p.poll() is None:
                self._kill_process_group(p, signal.SIGKILL)
                with contextlib.suppress(subprocess.TimeoutExpired):
                    p.wait(timeout=0.5)

    def _fail_child_died(self, name: str) -> None:
        print(
            f"\n{AnsiEsc.RED}✗{AnsiEsc.RESET} {name} process died unexpectedly",
            file=sys.stderr,
        )
        print(f"{AnsiEsc.DIM}Check {LOGFILE} for details{AnsiEsc.RESET}", file=sys.stderr)
        sys.exit(1)

    def wait_http_ready(
        self,
        *,
        checks: list[HealthCheck],
        name: str,
        port_for_tcp_hint: int,
        proc_getter: Callable[[], subprocess.Popen[str] | None],
        pid: int | None = None,
    ) -> None:
        """
        Wait until ANY check passes. Validates HTTP status codes (and optional body predicate).
        Also checks for child liveness while waiting.
        """
        start = time.time()
        last_error: str | None = None
        last_status: int | None = None
        last_url: str | None = None
        last_body_snip: str | None = None

        while time.time() < (start + self.startup_timeout_seconds):
            proc = proc_getter()
            if proc and proc.poll() is not None:
                self._fail_child_died(name)

            # TCP hint first to reduce noisy connect exceptions
            if not _tcp_open("localhost", port_for_tcp_hint, timeout=0.25):
                time.sleep(0.25)
                continue

            for hc in checks:
                try:
                    resp = self._session.get(
                        hc.url,
                        timeout=hc.timeout,
                        headers=hc.headers,
                        allow_redirects=hc.allow_redirects,
                    )
                    last_url = hc.url
                    last_status = resp.status_code
                    last_body_snip = resp.text[:200].replace("\n", "\\n")

                    if resp.status_code in hc.ok_statuses:
                        if hc.body_predicate and not hc.body_predicate(resp):
                            last_error = "body predicate failed"
                            continue

                        if pid is not None:
                            print(
                                f"  {AnsiEsc.GREEN}✓{AnsiEsc.RESET} {name} started {AnsiEsc.DIM}(pid {pid}){AnsiEsc.RESET}"
                            )
                        return

                    last_error = f"unexpected status {resp.status_code}"
                except requests.RequestException as e:
                    last_error = f"request error: {type(e).__name__}: {e}"

            time.sleep(0.4)

        # Timeout diagnostics
        print(f"{AnsiEsc.RED}✗{AnsiEsc.RESET} {name} healthcheck failed", file=sys.stderr)
        if last_url is not None:
            print(
                f"{AnsiEsc.DIM}Last check:{AnsiEsc.RESET} {last_url}",
                file=sys.stderr,
            )
        if last_status is not None:
            print(
                f"{AnsiEsc.DIM}Last status:{AnsiEsc.RESET} {last_status}",
                file=sys.stderr,
            )
        if last_error is not None:
            print(
                f"{AnsiEsc.DIM}Last error:{AnsiEsc.RESET} {last_error}",
                file=sys.stderr,
            )
        if last_body_snip:
            print(
                f"{AnsiEsc.DIM}Body snippet:{AnsiEsc.RESET} {last_body_snip}",
                file=sys.stderr,
            )
        print(f"{AnsiEsc.DIM}Check {LOGFILE} for details{AnsiEsc.RESET}", file=sys.stderr)
        sys.exit(1)

    def spawn_backend(self, port: int, logfile: TextIO) -> subprocess.Popen[str]:
        project_root = APP_DIR.parent.parent
        cmd = [
            "uv",
            "run",
            "python",
            "-u",
            str(APP_DIR / "backend" / "server.py"),
            "--port",
            str(port),
        ]
        proc = _spawn(cmd, cwd=project_root, env=None, logfile=logfile)
        self.backend_process = proc
        return proc

    def spawn_frontend(
        self, port: int, backend_port: int, logfile: TextIO
    ) -> subprocess.Popen[str]:
        env = os.environ.copy()
        env["BACKEND_URL"] = f"http://localhost:{backend_port}"
        cmd = ["npm", "run", "dev", "--", "--port", str(port), "--strictPort"]
        proc = _spawn(cmd, cwd=APP_DIR / "frontend", env=env, logfile=logfile)
        self.frontend_process = proc
        return proc

    def monitor_child_liveness(self) -> None:
        log_lines_to_show = 20
        prev_lines: list[str] = []

        while True:
            if self.backend_process and self.backend_process.poll() is not None:
                self._fail_child_died("Backend")
            if self.frontend_process and self.frontend_process.poll() is not None:
                self._fail_child_died("Frontend")

            # Show last N lines of logs in a box
            try:
                with open(LOGFILE) as f:
                    all_lines = f.readlines()
                tail = all_lines[-log_lines_to_show:]

                if tail != prev_lines:
                    # Clear previous log display (box has +2 lines for borders)
                    if prev_lines:
                        lines_to_clear = len(prev_lines) + 2
                        print(f"\033[{lines_to_clear}A\033[J", end="")

                    print(f"{AnsiEsc.DIM}┌─ logs {'─' * 32}{AnsiEsc.RESET}")
                    for line in tail:
                        print(f"{AnsiEsc.DIM}│ {line.rstrip()}{AnsiEsc.RESET}")
                    print(f"{AnsiEsc.DIM}└{'─' * 40}{AnsiEsc.RESET}")

                    prev_lines = tail
            except FileNotFoundError:
                pass

            time.sleep(1.0)

    def run(self) -> None:
        """Launch the backend and frontend development servers."""
        print(f"{AnsiEsc.DIM}Logfile: {LOGFILE}{AnsiEsc.RESET}")
        print(f"{AnsiEsc.DIM}Finding available ports...{AnsiEsc.RESET}")

        bport = find_available_port(BACKEND_DEFAULT_START)
        fport = find_available_port(FRONTEND_DEFAULT_START)

        print(f" - {AnsiEsc.DIM}Backend port: {bport}{AnsiEsc.RESET}")
        print(f" - {AnsiEsc.DIM}Frontend port: {fport}{AnsiEsc.RESET}\n")

        print(f"{AnsiEsc.BOLD}Starting development servers{AnsiEsc.RESET}")
        print(f"{AnsiEsc.DIM}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{AnsiEsc.RESET}")

        with open(LOGFILE, "a", buffering=1, encoding="utf-8") as logfile:
            check_host = "localhost"

            # Start backend first and wait for it to be ready
            print(f"  {AnsiEsc.DIM}▸ Spawning backend...{AnsiEsc.RESET}")
            backend_proc = self.spawn_backend(bport, logfile)

            backend_checks = [
                HealthCheck(
                    url=f"http://{check_host}:{bport}/api/health",
                    ok_statuses={200},
                    timeout=1.0,
                )
            ]

            self.wait_http_ready(
                checks=backend_checks,
                name="Backend",
                port_for_tcp_hint=bport,
                proc_getter=lambda: self.backend_process,
                pid=backend_proc.pid,
            )

            # Start frontend after backend is ready
            print(f"  {AnsiEsc.DIM}▸ Spawning frontend...{AnsiEsc.RESET}")
            frontend_proc = self.spawn_frontend(fport, bport, logfile)

            frontend_checks = [
                HealthCheck(
                    url=f"http://{check_host}:{fport}/",
                    ok_statuses={200, 204, 301, 302, 304},
                    timeout=1.0,
                    allow_redirects=True,
                ),
                HealthCheck(
                    url=f"http://{check_host}:{fport}/@vite/client",
                    ok_statuses={200, 304},
                    timeout=1.0,
                    allow_redirects=True,
                ),
            ]

            self.wait_http_ready(
                checks=frontend_checks,
                name="Frontend",
                port_for_tcp_hint=fport,
                proc_getter=lambda: self.frontend_process,
                pid=frontend_proc.pid,
            )

            print(f"{AnsiEsc.DIM}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{AnsiEsc.RESET}\n")
            time.sleep(0.1)

            print(
                f"{AnsiEsc.BOLD}Ready: {AnsiEsc.GREEN}{AnsiEsc.UNDERLINE}http://localhost:{fport}/{AnsiEsc.RESET}\n"
            )

            self.monitor_child_liveness()


def main() -> None:
    LOGFILE.unlink(missing_ok=True)
    with open(LOGFILE, "w", encoding="utf-8") as lf:
        lf.write(f"Launcher started at {datetime.now().isoformat()}\n")

    _require_bins("uv", "npm")

    runner = AppRunner(startup_timeout_seconds=DEFAULT_STARTUP_TIMEOUT_SECONDS)

    def signal_handler(_signum: int, _frame: FrameType | None) -> None:
        runner.cleanup()
        sys.exit(0)

    atexit.register(runner.cleanup)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGHUP, signal_handler)

    runner.run()


if __name__ == "__main__":
    main()
