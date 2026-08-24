from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from param_decomp.experiments.lm import deliverable


def _product_record(dataset_name: str = "train-v1") -> dict[str, Any]:
    return {
        "target": {
            "spec": {
                "kind": "hf",
                "model_class": "transformers.LlamaForCausalLM",
                "model_name": "meta-llama/Llama-3.1-8B",
            },
            "weights_dtype": "bfloat16",
            "attention_implementation": "auto",
        },
        "decomposition": {
            "sites": {
                "kind": "glu_transformer",
                "layers": {"kind": "range", "start": 0, "end": 1},
                "cs": {"q": 1},
            },
            "ci": {
                "type": "chunkwise_transformer",
                "blocks_per_chunk": 1,
                "input_tap": "first_block_resid",
                "d_model": 8,
                "n_blocks": 1,
                "attention": {"kind": "mha", "n_heads": 1},
                "ffn": {"kind": "gelu", "hidden": 8},
            },
        },
        "data": {
            "train": {"kind": "name", "name": dataset_name},
            "eval": {"kind": "dir", "dir": "/datasets/eval"},
        },
        "pd": {"seed": 7, "process_only": "opaque"},
    }


def _resolve(monkeypatch: pytest.MonkeyPatch) -> tuple[object, object]:
    resolved_target = object()
    resolved_ci = object()
    monkeypatch.setattr(
        deliverable,
        "resolve_decomposition",
        lambda *_args: SimpleNamespace(target=resolved_target, tree=object(), grammar=object()),
    )
    monkeypatch.setattr(deliverable, "resolve_lm_ci_arch", lambda *_args: resolved_ci)
    return resolved_target, resolved_ci


def test_current_launch_pin_resolves_product(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "p-current"
    run_dir.mkdir()
    (run_dir / "launch_config.yaml").write_text(yaml.safe_dump(_product_record()))
    resolved_target, resolved_ci = _resolve(monkeypatch)

    result = deliverable.load_deliverable(run_dir, tmp_path)

    assert result.target is resolved_target
    assert result.ci_fn is resolved_ci
    assert result.seed == 7
    assert result.data.dir == tmp_path / "datasets" / "train-v1"
    assert result.data.eval_dir == Path("/datasets/eval")


def test_normalized_product_takes_precedence_over_process_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "p-normalized"
    run_dir.mkdir()
    (run_dir / "launch_config.yaml").write_text(yaml.safe_dump({"historical": "opaque"}))
    (run_dir / deliverable.DELIVERABLE_FILENAME).write_text(
        yaml.safe_dump({**_product_record("normalized-v1"), "provenance": {"git_commit": "abc"}})
    )
    _resolve(monkeypatch)

    result = deliverable.load_deliverable(run_dir, tmp_path)

    assert result.data.dir == tmp_path / "datasets" / "normalized-v1"


def test_current_schema_rejects_retired_initialization(tmp_path: Path) -> None:
    record = _product_record()
    record["decomposition"]["sites"]["initialization"] = "neuron_head_aligned"
    run_dir = tmp_path / "p-retired"
    run_dir.mkdir()
    (run_dir / "launch_config.yaml").write_text(yaml.safe_dump(record))

    with pytest.raises(ValidationError):
        deliverable.load_deliverable(run_dir, tmp_path)
