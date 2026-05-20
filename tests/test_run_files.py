from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, TensorDataset

from param_decomp.adapters.base import pretrain_dataloader
from param_decomp.app.backend.routers.pretrain_info import _get_dataset_short
from param_decomp.pretrain.run_info import PretrainRunInfo
from param_decomp.utils import run_files


def test_resolve_wandb_run_files_uses_fixed_checkpoint_filename(
    monkeypatch: Any, tmp_path: Path
) -> None:
    class FakeRun:
        id = "abcdef12"

    class FakeApi:
        def run(self, wandb_path: str) -> FakeRun:
            assert wandb_path == "entity/project/abcdef12"
            return FakeRun()

    def fake_download_wandb_file(_run: FakeRun, run_dir: Path, file_name: str) -> Path:
        return run_dir / file_name

    def fail_fetch_latest_wandb_checkpoint(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("fixed checkpoint filenames should not use latest-checkpoint lookup")

    monkeypatch.setattr("param_decomp.utils.run_files.wandb.Api", FakeApi)
    monkeypatch.setattr(run_files, "PARAM_DECOMP_OUT_DIR", tmp_path)
    monkeypatch.setattr(run_files, "download_wandb_file", fake_download_wandb_file)
    monkeypatch.setattr(
        run_files, "fetch_latest_wandb_checkpoint", fail_fetch_latest_wandb_checkpoint
    )

    resolved = run_files.resolve_run_files(
        "wandb:entity/project/runs/abcdef12",
        config_filename="target_config.yaml",
        checkpoint_filename="target_model.pth",
        extras_from_config_path=lambda _path: ["extra.json"],
    )

    cache_dir = tmp_path / "runs" / "project-abcdef12"
    assert resolved.config_path == cache_dir / "target_config.yaml"
    assert resolved.checkpoint_path == cache_dir / "target_model.pth"
    assert resolved.extras == {"extra.json": cache_dir / "extra.json"}


def test_resolve_wandb_run_files_reuses_download_cache(monkeypatch: Any, tmp_path: Path) -> None:
    class FakeRun:
        id = "abcdef12"

    api_calls = 0

    class FakeApi:
        def run(self, wandb_path: str) -> FakeRun:
            nonlocal api_calls
            api_calls += 1
            assert wandb_path == "entity/project/abcdef12"
            return FakeRun()

    def fake_download_wandb_file(_run: FakeRun, run_dir: Path, file_name: str) -> Path:
        path = run_dir / file_name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(file_name)
        return path

    monkeypatch.setattr("param_decomp.utils.run_files.wandb.Api", FakeApi)
    monkeypatch.setattr(run_files, "PARAM_DECOMP_OUT_DIR", tmp_path)
    monkeypatch.setattr(run_files, "download_wandb_file", fake_download_wandb_file)

    first = run_files.resolve_run_files(
        "wandb:entity/project/runs/abcdef12",
        config_filename="target_config.yaml",
        checkpoint_filename="target_model.pth",
        extras_from_config_path=lambda _path: ["extra.json"],
    )
    second = run_files.resolve_run_files(
        "wandb:entity/project/runs/abcdef12",
        config_filename="target_config.yaml",
        checkpoint_filename="target_model.pth",
        extras_from_config_path=lambda _path: ["extra.json"],
    )

    cache_dir = tmp_path / "runs" / "project-abcdef12"
    assert first.config_path == second.config_path == cache_dir / "target_config.yaml"
    assert first.checkpoint_path == second.checkpoint_path == cache_dir / "target_model.pth"
    assert first.extras == second.extras == {"extra.json": cache_dir / "extra.json"}
    assert api_calls == 1


def test_pretrain_info_dataset_short_reads_new_data_config() -> None:
    assert _get_dataset_short({"data": {"dataset_name": "SimpleStories/SimpleStories"}}) == "SS"


def test_pretrain_dataloader_uses_model_block_size(monkeypatch: Any) -> None:
    captured: dict[str, Any] = {}

    def fake_create_lm_data_loader(
        cfg: Any,
        *,
        split: str,
        batch_size: int,
        seed: int,
        collate_fn: Any,
    ) -> tuple[DataLoader[Any], None]:
        captured.update(
            {
                "max_seq_len": cfg.max_seq_len,
                "split": split,
                "batch_size": batch_size,
                "seed": seed,
                "collate_fn": collate_fn,
            }
        )
        return DataLoader(TensorDataset(torch.empty(0, dtype=torch.long))), None

    monkeypatch.setattr(
        "param_decomp.experiments.lm.data.create_lm_data_loader",
        fake_create_lm_data_loader,
    )

    run_info = PretrainRunInfo(
        checkpoint_path=Path("model.pt"),
        config_dict={
            "seed": 123,
            "data": {
                "dataset_name": "SimpleStories/SimpleStories",
                "tokenizer_name": "tokenizer",
                "column_name": "story",
                "max_seq_len": 513,
                "train_split": "train",
                "eval_split": "test",
                "is_tokenized": False,
                "streaming": False,
            },
        },
        model_config_dict={"block_size": 512},
        tokenizer_path=None,
        hf_tokenizer_path="tokenizer",
        seed=123,
    )

    loader = pretrain_dataloader(run_info, batch_size=7)

    assert isinstance(loader, DataLoader)
    assert captured["max_seq_len"] == 512
    assert captured["split"] == "train"
    assert captured["batch_size"] == 7
    assert captured["seed"] == 123
