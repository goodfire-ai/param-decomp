from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

import pytest
import yaml
from pydantic import ValidationError

from param_decomp.experiments.lm import deliverable
from param_decomp.experiments.lm.config import LMDecompositionConfig, LMTargetConfig
from param_decomp.experiments.lm.run_data import DELIVERABLE_FILENAME


def _product_record(dataset_name: str = "train-v1") -> dict[str, Any]:
    return {
        "target": {
            "spec": {
                "kind": "hf",
                "model_class": "transformers.LlamaForCausalLM",
                "model_name": "meta-llama/Llama-3.1-8B",
            },
            "weights_dtype": "bfloat16",
            "output_edge": {"kind": "materialized"},
            "attention_implementation": "xla",
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
                "attention": {
                    "mask": "bidirectional",
                    "kind": "mha",
                    "implementation": "xla",
                    "n_heads": 1,
                },
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
    monkeypatch.setattr(deliverable, "resolve_lm_ci_fn_arch", lambda *_args: resolved_ci)
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


@pytest.mark.parametrize("policy", [None, "auto", "cudnn"])
@pytest.mark.parametrize("field", ["target", "ci"])
def test_stored_product_refuses_ambiguous_attention(
    tmp_path: Path, policy: str | None, field: Literal["target", "ci"]
) -> None:
    record = _product_record()
    match field:
        case "target":
            policy_owner = record["target"]
            policy_key = "attention_implementation"
        case "ci":
            policy_owner = record["decomposition"]["ci"]["attention"]
            policy_key = "implementation"
    if policy is None:
        del policy_owner[policy_key]
    else:
        policy_owner[policy_key] = policy
    run_dir = tmp_path / "p-ambiguous-attention"
    run_dir.mkdir()
    (run_dir / "launch_config.yaml").write_text(yaml.safe_dump(record))

    with pytest.raises(ValidationError, match=policy_key):
        deliverable.load_deliverable(run_dir, tmp_path)


@pytest.mark.parametrize("policy", ["flash", "xla"])
def test_target_schema_preserves_explicit_attention(policy: str) -> None:
    record = _product_record()
    record["target"]["attention_implementation"] = policy
    assert LMTargetConfig.model_validate(record["target"]).attention_implementation == policy


def test_normalized_product_takes_precedence_over_process_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "p-normalized"
    run_dir.mkdir()
    (run_dir / "launch_config.yaml").write_text(yaml.safe_dump({"historical": "opaque"}))
    (run_dir / DELIVERABLE_FILENAME).write_text(
        yaml.safe_dump({**_product_record("normalized-v1"), "provenance": {"git_commit": "abc"}})
    )
    _resolve(monkeypatch)

    result = deliverable.load_deliverable(run_dir, tmp_path)

    assert result.data.dir == tmp_path / "datasets" / "normalized-v1"


def test_current_schema_accepts_nonlinearity_aligned_initialization() -> None:
    record = _product_record()
    record["decomposition"]["sites"]["initialization"] = "nonlinearity_aligned"

    parsed = LMDecompositionConfig.model_validate(record["decomposition"])

    assert parsed.sites.kind == "glu_transformer"
    assert parsed.sites.initialization == "nonlinearity_aligned"


def test_target_schema_refuses_flash_float32():
    target = _product_record()["target"]
    target["weights_dtype"] = "float32"
    target["attention_implementation"] = "flash"
    with pytest.raises(ValidationError, match="flash attention requires bfloat16"):
        LMTargetConfig.model_validate(target)
    target["attention_implementation"] = "xla"
    assert LMTargetConfig.model_validate(target).weights_dtype == "float32"


def test_the_dense_target_resolves_without_reading_the_ci_definition(tmp_path: Path) -> None:
    record = _product_record()
    record["decomposition"]["ci"] = {"type": "a_ci_this_schema_never_had"}
    (tmp_path / DELIVERABLE_FILENAME).write_text(yaml.safe_dump(record))
    with pytest.raises(ValidationError):
        deliverable.load_deliverable(tmp_path, tmp_path)
    target = deliverable.load_dense_target(tmp_path, tmp_path)
    assert [site.name for site in target.sites] == ["layers.0.self_attn.q_proj"]


def test_the_dense_target_refuses_what_it_reads_but_does_not_know(tmp_path: Path) -> None:
    record = _product_record()
    record["decomposition"]["sites"]["a_field_this_schema_never_had"] = 1
    (tmp_path / DELIVERABLE_FILENAME).write_text(yaml.safe_dump(record))
    with pytest.raises(ValidationError):
        deliverable.load_dense_target(tmp_path, tmp_path)
