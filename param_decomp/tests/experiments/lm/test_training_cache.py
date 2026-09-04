"""`runtime.compilation_cache_dir` is authored, never ambient: the schema requires it,
the seats author the per-user home (XLA's autotune subdir is not safe for unrelated
Unix users to share), and the trainer only `~`-expands what was written."""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from param_decomp.experiments.lm.runtime import RuntimeConfig
from param_decomp.tests.experiments.lm.test_runtime import _MINIMAL_RUNTIME

CONFIGS = Path(__file__).parents[3] / "experiments" / "lm" / "configs"


def test_compilation_cache_dir_is_required():
    absent = {k: v for k, v in _MINIMAL_RUNTIME.items() if k != "compilation_cache_dir"}
    with pytest.raises(ValidationError, match="compilation_cache_dir"):
        RuntimeConfig.model_validate(absent)


def test_authored_tilde_expands_to_the_running_users_home():
    runtime = RuntimeConfig.model_validate(_MINIMAL_RUNTIME)
    assert runtime.compilation_cache_dir == Path("~/.cache/param-decomp/xla")
    expanded = runtime.compilation_cache_dir.expanduser()
    assert expanded.is_absolute() and expanded == Path.home() / ".cache/param-decomp/xla"


def test_every_seat_authors_a_per_user_cache_dir():
    """Per-user isolation is the seats' AUTHORED value, not a code default — a seat
    pointing the cache under a shared artifact root reintroduces the cross-user autotune
    collision."""
    for seat in sorted(CONFIGS.glob("*.yaml")):
        authored = yaml.safe_load(seat.read_text())["runtime"]["compilation_cache_dir"]
        assert isinstance(authored, str) and authored.startswith("~/"), (seat.name, authored)
