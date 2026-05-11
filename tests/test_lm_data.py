from typing import Any

from param_decomp.experiments.lm.configs import LMDataConfig
from param_decomp.experiments.lm.data import build_lm_dataloaders


def test_build_lm_dataloaders_uses_caller_seed(monkeypatch: Any) -> None:
    calls: list[dict[str, Any]] = []

    def fake_create_data_loader(**kwargs: Any) -> tuple[list[Any], None]:
        calls.append(kwargs)
        return [], None

    monkeypatch.setattr(
        "param_decomp.experiments.lm.data.create_data_loader",
        fake_create_data_loader,
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

    assert calls[0]["dataset_config"].seed == 99
    assert calls[0]["global_seed"] == 99
    assert calls[1]["dataset_config"].seed == 100
    assert calls[1]["global_seed"] == 100
