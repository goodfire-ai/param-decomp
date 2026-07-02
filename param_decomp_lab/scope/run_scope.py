"""Dev launcher for scope: backend on a free port, then the SvelteKit dev server proxying /api."""

import argparse
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

SCOPE_DIR = Path(__file__).parent.resolve()
FRONTEND_DIR = SCOPE_DIR / "frontend"
FIXTURE_STORE = Path(tempfile.gettempdir()) / "scope-fixture-store"
STARTUP_TIMEOUT_S = 60


def seed_fixture_store() -> None:
    """Point PARAM_DECOMP_OUT_DIR at a throwaway store and write synthetic shards into it
    (once) so the backend's real read path has something to serve in dev."""
    os.environ["PARAM_DECOMP_OUT_DIR"] = str(FIXTURE_STORE)
    if (FIXTURE_STORE / "runs").exists():
        return
    from param_decomp_lab.scope.fixture import write_fixture_store

    print(f"seeding fixture store at {FIXTURE_STORE} ...")
    write_fixture_store()


def find_free_port(start: int) -> int:
    for port in range(start, start + 100):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError(f"no free port in [{start}, {start + 100})")


def http_ok(url: str) -> bool:
    request = urllib.request.Request(url, headers={"Accept": "text/html,application/json"})
    try:
        with urllib.request.urlopen(request, timeout=2) as response:
            return response.status == 200
    except (urllib.error.URLError, TimeoutError):
        return False


def wait_until_serving(name: str, url: str, process: subprocess.Popen[bytes]) -> None:
    deadline = time.time() + STARTUP_TIMEOUT_S
    while time.time() < deadline:
        assert process.poll() is None, f"{name} exited with code {process.returncode}"
        if http_ok(url):
            print(f"  {name} serving at {url}")
            return
        time.sleep(0.3)
    raise RuntimeError(f"{name} did not serve {url} within {STARTUP_TIMEOUT_S}s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the scope backend + frontend dev servers")
    parser.add_argument("--backend-port", type=int, default=None)
    parser.add_argument("--frontend-port", type=int, default=None)
    args = parser.parse_args()

    assert shutil.which("npm") is not None, "npm is required to run the scope frontend"
    if not (FRONTEND_DIR / "node_modules").exists():
        print("installing frontend dependencies (first run)...")
        subprocess.run(["npm", "install"], cwd=FRONTEND_DIR, check=True)

    seed_fixture_store()

    backend_port = args.backend_port or find_free_port(8000)
    frontend_port = args.frontend_port or find_free_port(5173)
    backend_url = f"http://127.0.0.1:{backend_port}"

    backend = subprocess.Popen(
        [sys.executable, "-m", "param_decomp_lab.scope.backend.server", "--port", str(backend_port)]
    )
    frontend: subprocess.Popen[bytes] | None = None
    try:
        wait_until_serving("backend", f"{backend_url}/api/catalog", backend)

        frontend = subprocess.Popen(
            ["npm", "run", "dev", "--", "--port", str(frontend_port), "--strictPort"],
            cwd=FRONTEND_DIR,
            env=os.environ | {"SCOPE_BACKEND_URL": backend_url},
        )
        wait_until_serving("frontend", f"http://127.0.0.1:{frontend_port}/", frontend)

        print(f"\n  scope: http://localhost:{frontend_port}  (backend {backend_url})\n")
        backend.wait()
    finally:
        for process in (frontend, backend):
            if process is not None and process.poll() is None:
                process.terminate()
        for process in (frontend, backend):
            if process is not None:
                process.wait(timeout=10)


if __name__ == "__main__":
    main()
