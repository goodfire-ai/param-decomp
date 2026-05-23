import pytest

from param_decomp_lab.infra.wandb import parse_wandb_run_path


@pytest.mark.parametrize(
    "input_path, expected",
    [
        ("myentity/myproject/abcd1234", ("myentity", "myproject", "abcd1234")),
        ("goodfire/spd/runs/xy7z9abc", ("goodfire", "spd", "xy7z9abc")),
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
        (
            "https://wandb.ai/goodfire/spd/runs/s-d2ec3bfe",
            ("goodfire", "spd", "s-d2ec3bfe"),
        ),
    ],
    ids=[
        "compact",
        "compact-with-runs",
        "url",
        "url-with-query",
        "whitespace",
        "special-chars",
        "hyphenated-runid",
        "hyphenated-runid-url",
    ],
)
def test_parse_wandb_run_path_valid(input_path: str, expected: tuple[str, str, str]):
    assert parse_wandb_run_path(input_path) == expected


@pytest.mark.parametrize(
    "input_path",
    [
        "/path/to/abcd1234",
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
        "wandb:myentity/myproject/abcd1234",
        "wandb:goodfire/spd/runs/s-d2ec3bfe",
    ],
    ids=[
        "is-an-absolute-path",
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
        "legacy-wandb-prefix-compact",
        "legacy-wandb-prefix-with-runs",
    ],
)
def test_parse_wandb_run_path_invalid(input_path: str):
    with pytest.raises(ValueError, match="Invalid W&B run reference"):
        parse_wandb_run_path(input_path)
