"""Add runslow option and skip slow tests if not specified.

Taken from https://docs.pytest.org/en/latest/example/simple.html.
"""

import os
import sys
from collections.abc import Iterable, Iterator
from netrc import netrc
from pathlib import Path
from urllib.parse import urlparse

import pytest
from pytest import Config, Item, Parser

# Must land before the first `import jax` anywhere in the suite (jax reads XLA_FLAGS at
# backend init) — merge rather than overwrite, since `make test-multidevice` sets its own
# XLA_FLAGS. Caps codegen at AVX2 so a cached executable stays valid across any x86-64-v3+
# host, not just the one that compiled it: an AVX512-tier compile embeds a target-cpu guess
# that can mismatch the loading host's feature set (harmless-looking warning, but "harmless"
# isn't good enough for a shared CI cache — this makes cross-host reuse safe by construction).
_MAX_ISA_FLAG = "--xla_cpu_max_isa=AVX2"
if _MAX_ISA_FLAG not in os.environ.get("XLA_FLAGS", ""):
    os.environ["XLA_FLAGS"] = f"{os.environ.get('XLA_FLAGS', '')} {_MAX_ISA_FLAG}".strip()

import jax  # noqa: E402

# Compiles are keyed by HLO + backend + jax/xla version, so this only ever serves a bit-exact
# rerun of the same compile; a threshold of 1s skips caching the hundreds of sub-second
# internal jits (which would otherwise bloat the dir and spam load-time log lines) while
# still catching every test-suite compile worth caching (the slow ones run 4-20s). One dir
# per xdist worker: jax's own writer-gating (`process_id == 0`) is a no-op here since each
# worker is an independent single-process jax runtime that trivially IS "process 0" — a
# shared dir would let concurrent workers race on the same cache files, so give each its own.
_cache_root = Path(__file__).parent / ".jax_test_cache"
_worker_cache_dir = _cache_root / os.environ.get("PYTEST_XDIST_WORKER", "master")
jax.config.update("jax_compilation_cache_dir", str(_worker_cache_dir))
jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)

# Each live XLA CPU executable consumes several non-mergeable VMAs. A serial suite can
# exhaust Linux's `vm.max_map_count`, whereupon XLA aborts instead of raising. Clear at
# module boundaries once half full, leaving room for `test_checkpoint.py`'s ~19k mappings.
_MAPPING_CEILING = (
    int(Path("/proc/sys/vm/max_map_count").read_text()) if sys.platform == "linux" else None
)


def _mapping_count() -> int:
    return Path("/proc/self/maps").read_bytes().count(b"\n")


@pytest.fixture(autouse=True, scope="module")
def _bounded_jax_executable_mappings() -> Iterator[None]:
    yield
    if _MAPPING_CEILING is not None and _mapping_count() > _MAPPING_CEILING // 2:
        jax.clear_caches()


def pytest_addoption(parser: Parser) -> None:
    parser.addoption("--runslow", action="store_true", default=False, help="run slow tests")
    parser.addoption(
        "--runmultidevice",
        action="store_true",
        default=False,
        help="run tests needing >1 jax device (requires XLA_FLAGS=--xla_force_host_platform_device_count>=2; "
        "use `make test-multidevice`)",
    )


def pytest_configure(config: Config) -> None:
    config.addinivalue_line("markers", "slow: mark test as slow to run")
    config.addinivalue_line("markers", "requires_wandb: mark test as requiring WANDB credentials")
    config.addinivalue_line(
        "markers",
        "multidevice: needs >1 jax device; hangs at the default 1 device, so skipped unless "
        "--runmultidevice (use `make test-multidevice`, which sets XLA_FLAGS for simulated CPU devices)",
    )


def _wandb_host() -> str:
    base_url = os.environ.get("WANDB_BASE_URL", "https://api.wandb.ai")
    parsed = urlparse(base_url)
    host = parsed.netloc or parsed.path or "api.wandb.ai"
    return host.split("/")[0]


def _have_wandb_credentials() -> bool:
    """Check if we have WANDB credentials.

    We check for either of:
    - WANDB_API_KEY environment variable
    - .netrc file in the home directory
    """

    if os.environ.get("WANDB_API_KEY"):
        return True
    host = _wandb_host()
    netrc_path = Path.home() / ".netrc"
    if not netrc_path.exists():
        return False
    try:
        n = netrc(netrc_path.as_posix())
        return n.authenticators(host) is not None
    except Exception:
        return False


def pytest_collection_modifyitems(config: Config, items: Iterable[Item]) -> None:
    runslow = config.getoption("--runslow")
    runmultidevice = config.getoption("--runmultidevice")
    have_wandb = _have_wandb_credentials()
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    skip_multidevice = pytest.mark.skip(reason="needs >1 device; run via `make test-multidevice`")
    skip_wandb = pytest.mark.skip(
        reason="No WANDB credentials (set WANDB_API_KEY or login via CLI)"
    )
    for item in items:
        if "slow" in item.keywords and not runslow:
            item.add_marker(skip_slow)
        if "multidevice" in item.keywords and not runmultidevice:
            item.add_marker(skip_multidevice)
        if "requires_wandb" in item.keywords and not have_wandb:
            item.add_marker(skip_wandb)
