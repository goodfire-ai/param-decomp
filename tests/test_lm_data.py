from typing import Any

from param_decomp.experiments.lm.data import LMDataConfig, build_lm_dataloaders


def test_build_lm_dataloaders_uses_explicit_seed_override(monkeypatch: Any) -> None:
    calls: list[dict[str, Any]] = []

    def fake_create_lm_data_loader(**kwargs: Any) -> tuple[list[Any], None]:
        calls.append(kwargs)
        return [], None

    monkeypatch.setattr(
        "param_decomp.experiments.lm.data.create_lm_data_loader",
        fake_create_lm_data_loader,
    )

    data_cfg = LMDataConfig(
        dataset_name="dataset",
        tokenizer_name="tokenizer",
        column_name="text",
        max_seq_len=8,
        train_split="train",
        eval_split="eval",
        buffer_size=10,
        dataset_seed=1234,
    )

    build_lm_dataloaders(
        data_cfg,
        seed=99,
        train_batch_size=2,
        eval_batch_size=2,
        dist_state=None,
    )

    assert calls[0]["seed"] == 99
    assert calls[0]["split"] == "train"
    assert calls[1]["seed"] == 100
    assert calls[1]["split"] == "eval"


def test_build_lm_dataloaders_uses_configured_dataset_seed(monkeypatch: Any) -> None:
    calls: list[dict[str, Any]] = []

    def fake_create_lm_data_loader(**kwargs: Any) -> tuple[list[Any], None]:
        calls.append(kwargs)
        return [], None

    monkeypatch.setattr(
        "param_decomp.experiments.lm.data.create_lm_data_loader",
        fake_create_lm_data_loader,
    )

    data_cfg = LMDataConfig(
        dataset_name="dataset",
        tokenizer_name="tokenizer",
        dataset_seed=1234,
    )

    build_lm_dataloaders(
        data_cfg,
        seed=None,
        default_seed=99,
        train_batch_size=2,
        eval_batch_size=2,
        dist_state=None,
    )

    assert calls[0]["seed"] == 1234
    assert calls[1]["seed"] == 1235


def test_build_lm_dataloaders_falls_back_to_default_seed(monkeypatch: Any) -> None:
    calls: list[dict[str, Any]] = []

    def fake_create_lm_data_loader(**kwargs: Any) -> tuple[list[Any], None]:
        calls.append(kwargs)
        return [], None

    monkeypatch.setattr(
        "param_decomp.experiments.lm.data.create_lm_data_loader",
        fake_create_lm_data_loader,
    )

    data_cfg = LMDataConfig(dataset_name="dataset", tokenizer_name="tokenizer")

    build_lm_dataloaders(
        data_cfg,
        seed=None,
        default_seed=99,
        train_batch_size=2,
        eval_batch_size=2,
        dist_state=None,
    )

    assert calls[0]["seed"] == 99
    assert calls[1]["seed"] == 100
