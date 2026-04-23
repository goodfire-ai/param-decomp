"""Test loading models from wandb runs.

If these tests fail, you should consider making your changes backwards compatible so the tests pass.
If you're willing to make breaking changes, see spd/scripts/run.py for creating new runs with
the canonical configs, and update the registry with your new run(s).
"""

import pytest

from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.registry import EXPERIMENT_REGISTRY
from spd.utils.wandb_utils import parse_wandb_run_path

CANONICAL_EXPS = [
    (exp_name, exp_config.canonical_run)
    for exp_name, exp_config in EXPERIMENT_REGISTRY.items()
    if exp_config.canonical_run is not None
]


@pytest.mark.requires_wandb
@pytest.mark.slow
@pytest.mark.parametrize("exp_name, canonical_run", CANONICAL_EXPS)
def test_loading_from_wandb(exp_name: str, canonical_run: str) -> None:
    try:
        run_info = SPDRunInfo.from_path(canonical_run)
        ComponentModel.from_run_info(run_info)
    except Exception as e:
        e.add_note(f"Error loading {exp_name} from {canonical_run}")
        raise e


@pytest.mark.parametrize(
    "input_path, expected",
    [
        ("myentity/myproject/abcd1234", ("myentity", "myproject", "abcd1234")),
        ("goodfire/spd/runs/xy7z9abc", ("goodfire", "spd", "xy7z9abc")),
        ("wandb:myentity/myproject/abcd1234", ("myentity", "myproject", "abcd1234")),
        ("wandb:myentity/myproject/runs/abcd1234", ("myentity", "myproject", "abcd1234")),
        (
            "https://wandb.ai/myentity/myproject/runs/abcd1234",
            ("myentity", "myproject", "abcd1234"),
        ),
        (
            "https://wandb.ai/myentity/myproject/runs/abcd1234?workspace=user",
            ("myentity", "myproject", "abcd1234"),
        ),
        ("  myentity/myproject/abcd1234  ", ("myentity", "myproject", "abcd1234")),  # whitespace
        ("my-entity/my_project/abcd1234", ("my-entity", "my_project", "abcd1234")),  # special chars
        ("goodfire/spd/runs/s-d2ec3bfe", ("goodfire", "spd", "s-d2ec3bfe")),  # Newer runid format
        ("wandb:goodfire/spd/s-d2ec3bfe", ("goodfire", "spd", "s-d2ec3bfe")),
        (
            "https://wandb.ai/goodfire/spd/runs/s-d2ec3bfe",
            ("goodfire", "spd", "s-d2ec3bfe"),
        ),
    ],
    ids=[
        "compact",
        "compact-with-runs",
        "prefix-compact",
        "prefix-with-runs",
        "url",
        "url-with-query",
        "whitespace",
        "special-chars",
        "hyphenated-runid",
        "hyphenated-runid-prefix",
        "hyphenated-runid-url",
    ],
)
def test_parse_wandb_run_path_valid(input_path: str, expected: tuple[str, str, str]):
    assert parse_wandb_run_path(input_path) == expected


@pytest.mark.parametrize(
    "input_path",
    [
        "myentity/myproject",
        "myentity/myproject/abc1234",
        "myentity/myproject/abcd12345",
        "myentity/myproject/ABCD1234",
        "myentity/myproject/abcd_1234",
        "myentity/myproject/ab-cd1234",
        "abcd1234",
        "",
        "not a valid path at all",
        "http://wandb.ai/myentity/myproject/runs/abcd1234",
        "https://example.com/myentity/myproject/runs/abcd1234",
        "https://wandb.ai/myentity/myproject/abcd1234",
    ],
    ids=[
        "missing-runid",
        "runid-too-short",
        "runid-too-long",
        "runid-uppercase",
        "runid-has-underscore",
        "runid-prefix-too-long",
        "only-runid",
        "empty",
        "random-text",
        "http-not-https",
        "wrong-domain",
        "url-missing-runs",
    ],
)
def test_parse_wandb_run_path_invalid(input_path: str):
    with pytest.raises(ValueError, match="Invalid W&B run reference"):
        parse_wandb_run_path(input_path)
