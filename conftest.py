"""Add runslow option and skip slow tests if not specified.

Taken from https://docs.pytest.org/en/latest/example/simple.html.
"""

import os
from collections.abc import Iterable
from netrc import netrc
from pathlib import Path
from urllib.parse import urlparse

import pytest
from pytest import Config, Item, Parser


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
