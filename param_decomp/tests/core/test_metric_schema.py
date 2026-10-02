import io
import json
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

import numpy as np
import pytest
import wandb
import yaml
from PIL import Image
from pydantic import ValidationError

from param_decomp.core.built_run import RunInstance
from param_decomp.core.configs import WandbConfig
from param_decomp.core.metrics import BarChart, LineChart, PNGImage
from param_decomp.core.run import DeferredMediaRecord, MetricsSink
from param_decomp.core.run_files import LAUNCH_CONFIG_FILENAME
from param_decomp.metric_schema import (
    MetricNames,
    MetricSchema,
    recorded_metric_schema,
    validate_resume_schema,
)
from param_decomp.pretrain.config import PretrainConfig, PretrainWandbConfig
from param_decomp.pretrain.train import MetricsSink as PretrainMetricsSink


class FakeWandb(ModuleType):
    def __init__(self) -> None:
        super().__init__("wandb")
        self.logged: list[tuple[dict[str, object], int | None, bool | None]] = []
        self.defined: list[tuple[str, str]] = []
        self.plot = Mock()
        self.Table = Mock()
        self.Table.MAX_ROWS = wandb.Table.MAX_ROWS

    def log(
        self,
        payload: Mapping[str, object],
        step: int | None = None,
        commit: bool | None = None,
    ) -> None:
        self.logged.append((dict(payload), step, commit))

    def define_metric(self, name: str, step_metric: str) -> None:
        self.defined.append((name, step_metric))

    def Image(self, image: Image.Image) -> Image.Image:  # noqa: N802
        image.load()
        return image


def png() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (2, 3), (24, 48, 96)).save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.mark.parametrize("schema", ["legacy", "grouped"])
def test_scalar_names_change_only_at_wandb_boundary(tmp_path: Path, schema: MetricSchema):
    fake = FakeWandb()
    path = tmp_path / "metrics.jsonl"
    with path.open("w") as jsonl:
        sink = MetricsSink(jsonl, fake, schema)
        sink.log(37, {"total": 2.5, "faith": 1.25, "src_lr": 0.003})
        sink.log(41, {"total": 2.0})

    assert [json.loads(line) for line in path.read_text().splitlines()] == [
        {
            "step": 37,
            "train/loss/total": 2.5,
            "train/loss/FaithfulnessLoss": 1.25,
            "train/schedules/lr/src": 0.003,
        },
        {"step": 41, "train/loss/total": 2.0},
    ]
    match schema:
        case "legacy":
            first = {
                "train/loss/total": 2.5,
                "train/loss/FaithfulnessLoss": 1.25,
                "train/schedules/lr/src": 0.003,
            }
            second = {"train/loss/total": 2.0}
        case "grouped":
            first = {
                "train-loss/total": 2.5,
                "train-loss/faithfulness": 1.25,
                "train-lr/ppgd-src": 0.003,
            }
            second = {"train-loss/total": 2.0}
    assert fake.logged == [(first, 37, True), (second, 41, True)]


def test_synchronous_media_keep_contents_under_grouped_names(tmp_path: Path):
    fake = FakeWandb()
    encoded = png()
    with (tmp_path / "metrics.jsonl").open("w") as jsonl:
        sink = MetricsSink(jsonl, fake, "grouped")
        sink.log(
            80,
            {
                "eval/l0/bar_chart": BarChart((("q", 2.0), ("k", 3.0)), "site", "l0", "L0"),
                "eval/l0/per_position": LineChart(
                    np.array([0, 1]), (("q", np.array([0.5, 0.25])),), "position", "CI"
                ),
                "slow_eval/figures/causal_importance_values": PNGImage(encoded),
            },
        )
    assert (tmp_path / "metrics.jsonl").read_text() == '{"step": 80}\n'
    payload, step, commit = fake.logged[0]
    assert (step, commit) == (80, True)
    assert set(payload) == {
        "eval-ci-figures/l0-site-and-group-counts",
        "eval-ci-figures/l0-per-position",
        "eval-ci-figures/value-histogram-lower-leaky",
    }
    assert payload["eval-ci-figures/l0-site-and-group-counts"] is fake.plot.bar.return_value
    assert payload["eval-ci-figures/l0-per-position"] is fake.plot.line_series.return_value
    fake.Table.assert_called_once_with(columns=["site", "l0"], data=[["q", 2.0], ["k", 3.0]])
    fake.plot.bar.assert_called_once_with(fake.Table.return_value, "site", "l0", title="L0")
    fake.plot.line_series.assert_called_once_with(
        xs=[0, 1], ys=[[0.5, 0.25]], keys=["q"], title="CI", xname="position"
    )
    image = payload["eval-ci-figures/value-histogram-lower-leaky"]
    assert isinstance(image, Image.Image)
    assert image.size == (2, 3)
    assert image.getpixel((0, 0)) == (24, 48, 96)


@pytest.mark.parametrize("schema", ["legacy", "grouped"])
def test_deferred_snapshots_keep_semantic_axis_and_may_arrive_late(
    tmp_path: Path, schema: MetricSchema
):
    fake = FakeWandb()
    source = "slow_eval/figures/causal_importance_values"
    with (tmp_path / "metrics.jsonl").open("w") as jsonl:
        sink = MetricsSink(jsonl, fake, schema)
        sink.log(100, {"total": 2.0})
        for snapshot_step in (50, 0, 100):
            sink.log_deferred_media(
                DeferredMediaRecord("slow_eval/figure_step", snapshot_step, {source: png()})
            )
    match schema:
        case "legacy":
            axis, metric = "slow_eval/figure_step", source
        case "grouped":
            axis, metric = "axes/slow-figures-step", "eval-ci-figures/value-histogram-lower-leaky"
    assert fake.defined == [(metric, axis)]
    assert [record[axis] for record, _, _ in fake.logged[1:]] == [50.0, 0.0, 100.0]
    assert all(step is None and commit is False for _, step, commit in fake.logged[1:])
    assert all(set(record) == {axis, metric} for record, _, _ in fake.logged[1:])


def test_grouped_collision_is_detected_across_separate_records():
    names = MetricNames("grouped")
    first = {"train/loss/custom_loss": 1.5}
    assert names.record(first) == {"train-loss/custom-loss": 1.5}
    assert names.record(first) == {"train-loss/custom-loss": 1.5}
    with pytest.raises(ValueError, match="collision"):
        names.record({"train/loss/custom-loss": 2.5})


def test_legacy_allows_unclassified_keys_and_grouped_rejects_them():
    assert MetricNames("legacy").record({"new/diagnostic": 2.0}) == {"new/diagnostic": 2.0}
    with pytest.raises(ValueError, match="Unclassified metric"):
        MetricNames("grouped").record({"new/diagnostic": 2.0})


@pytest.mark.parametrize("config_type", [WandbConfig, PretrainWandbConfig])
def test_schema_config_is_opt_in_and_closed(config_type: type[WandbConfig | PretrainWandbConfig]):
    assert config_type(project="test").metric_schema == "legacy"
    assert config_type(project="test", metric_schema="grouped").metric_schema == "grouped"
    with pytest.raises(ValidationError, match="metric_schema"):
        config_type.model_validate({"project": "test", "metric_schema": "typo"})


@pytest.mark.parametrize("previous,requested", [("legacy", "grouped"), ("grouped", "legacy")])
def test_resume_rejects_both_schema_changes(previous: MetricSchema, requested: MetricSchema):
    config = {"wandb": {"metric_schema": previous}}
    with pytest.raises(ValueError, match="Cannot resume"):
        validate_resume_schema(config, requested)
    assert config == {"wandb": {"metric_schema": previous}}


@pytest.mark.parametrize(
    "config,expected",
    [
        ({}, "legacy"),
        ({"wandb": {}}, "legacy"),
        ({"wandb": {"project": "test", "metric_schema": "legacy"}}, "legacy"),
        ({"wandb": {"project": "test", "metric_schema": "grouped"}}, "grouped"),
    ],
)
def test_recorded_schema_including_old_configs(config: dict[str, object], expected: MetricSchema):
    assert recorded_metric_schema(config) == expected
    validate_resume_schema(config, expected)


@pytest.mark.parametrize(
    "settings", [None, [], "legacy", {"metric_schema": None}, {"metric_schema": "typo"}]
)
def test_malformed_saved_schema_is_rejected(settings: object):
    with pytest.raises(ValidationError):
        recorded_metric_schema({"wandb": settings})


def test_old_run_without_schema_cannot_resume_grouped():
    with pytest.raises(ValueError, match="logged with 'legacy'"):
        validate_resume_schema({}, "grouped")


@pytest.mark.parametrize("resumed", [False, True])
@pytest.mark.parametrize("schema", ["legacy", "grouped"])
def test_pd_run_schema_is_checked_before_publishing_launch_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, resumed: bool, schema: MetricSchema
):
    tracker = Mock(resumed=resumed)
    tracker.config.as_dict.return_value = {}
    initialize = Mock(return_value=tracker)
    save = Mock()
    define_metric = Mock()
    monkeypatch.setattr(wandb, "init", initialize)
    monkeypatch.setattr(wandb, "save", save)
    monkeypatch.setattr(wandb, "define_metric", define_metric)
    run = RunInstance(
        run_name="schema-test",
        run_id="p-12345678",
        out_dir=tmp_path,
        wandb=WandbConfig(project="test", metric_schema=schema),
        resume_provenance=None,
    )
    run.run_dir.mkdir()
    launch_config = {"wandb": {"project": "test", "metric_schema": schema}}
    launch_path = run.run_dir / LAUNCH_CONFIG_FILENAME
    launch_path.write_text(yaml.safe_dump(launch_config))

    match resumed, schema:
        case True, "grouped":
            with pytest.raises(ValueError, match="Cannot resume"):
                MetricsSink.for_run(run, is_main=True)
            tracker.config.update.assert_not_called()
            save.assert_not_called()
            define_metric.assert_not_called()
            tracker.finish.assert_not_called()
            assert not (run.run_dir / "metrics.jsonl").exists()
        case (False, "legacy") | (False, "grouped") | (True, "legacy"):
            sink = MetricsSink.for_run(run, is_main=True)
            assert sink._jsonl is not None
            sink._jsonl.close()
            if resumed:
                tracker.config.update.assert_not_called()
            else:
                tracker.config.update.assert_called_once_with(launch_config, allow_val_change=False)
            save.assert_called_once_with(str(launch_path), base_path=str(run.run_dir), policy="now")
            match schema:
                case "legacy":
                    define_metric.assert_not_called()
                case "grouped":
                    define_metric.assert_called_once_with("axes/*", hidden=True)
    initialize.assert_called_once_with(
        project="test",
        entity=None,
        name="schema-test",
        id="p-12345678",
        group=None,
        tags=[],
        resume="allow",
    )


@pytest.mark.parametrize("schema", ["legacy", "grouped"])
@pytest.mark.parametrize("resumed", [False, True])
def test_pretrain_names_change_only_at_wandb_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, schema: MetricSchema, resumed: bool
):
    cfg_path = Path(__file__).parents[2] / "pretrain/configs/gpt2_simple-2L.yaml"
    data = PretrainConfig.from_file(cfg_path).model_dump()
    data.update(
        data_root=tmp_path, run_id="schema-test", wandb={"project": "test", "metric_schema": schema}
    )
    cfg = PretrainConfig.model_validate(data)
    cfg.paths.run_dir.mkdir(parents=True)
    run = Mock(resumed=resumed)
    run.config.as_dict.return_value = {"wandb": {"metric_schema": schema}}
    initialize = Mock(return_value=run)
    finish = Mock()
    fake = FakeWandb()
    monkeypatch.setattr(wandb, "init", initialize)
    monkeypatch.setattr(wandb, "log", fake.log)
    monkeypatch.setattr(wandb, "finish", finish)
    sink = PretrainMetricsSink(cfg, is_main=True)
    record = {"train_loss": 2.0, "val_loss": 2.5, "lr": 0.003, "step_time_s": 0.2}
    try:
        sink.log(13, record)
    finally:
        sink.finish()
    assert json.loads((cfg.paths.run_dir / "metrics.jsonl").read_text()) == {"step": 13, **record}
    match schema:
        case "legacy":
            expected = record
        case "grouped":
            expected = {
                "pretrain-loss/train": 2.0,
                "pretrain-loss/validation": 2.5,
                "pretrain-schedules/lr": 0.003,
                "pretrain-performance/step-time-s": 0.2,
            }
    assert fake.logged == [(expected, 13, None)]
    initialize.assert_called_once()
    if resumed:
        run.config.update.assert_not_called()
    else:
        run.config.update.assert_called_once_with(
            cfg.model_dump(mode="json"), allow_val_change=False
        )
    finish.assert_called_once()
