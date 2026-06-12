"""LM experiment config schema: target spec, data settings, and the full YAML tree.

Runtime builders (`build_target`, `build_lm_loader`, `make_run_batch`) live in
`param_decomp_lab/experiments/lm/`.
"""

from typing import Annotated, Literal

from pydantic import Discriminator, Field, PositiveInt

from param_decomp_config.base import BaseConfig
from param_decomp_config.experiment import ExperimentConfig


class HFTarget(BaseConfig):
    """Load a HuggingFace model via `<model_class>.from_pretrained(<model_name>)`."""

    kind: Literal["hf"] = "hf"
    model_class: str
    model_name: str


class PretrainedTarget(BaseConfig):
    """Load an in-repo lab-pretrained model.

    `run_path` accepts any form `PretrainRunInfo.from_path` does — compact W&B
    (`entity/project/runId`), full W&B (`entity/project/runs/runId`), or a local
    checkpoint path (repo-relative paths are resolved at load time by `build_target`).
    """

    kind: Literal["pretrained"] = "pretrained"
    model_class: str
    run_path: str


class HFWeightsInVendored(BaseConfig):
    """Load HF pretrained weights into a vendored `param_decomp_lab.experiments.lm.pretrain.models.*`
    architecture via `<class>.from_hf_pretrained(<hub_id>)`.

    Useful when the decomposition target needs structural changes vs HF — e.g.
    `GPT2Simple`'s separate q/k/v projections vs HF's fused `c_attn`.
    """

    kind: Literal["hf_weights_in_vendored"] = "hf_weights_in_vendored"
    model_class: str  # must expose `from_hf_pretrained`
    model_name: str  # HF hub id


class RandomWeightsInVendored(BaseConfig):
    """Random-init a vendored architecture at the shapes of an HF model's config, via
    `<class>.from_hf_config_random(<hub_id>)`. Reads only the tiny cached config json —
    no weight files touch the filesystem.

    Benchmarking scaffolding: FLOP- and shape-identical to the pretrained target, so
    throughput/memory probes run without the N-rank snapshot load (the host-OOM /
    NFS-stall class). Never use for runs whose losses are meant to mean anything.
    """

    kind: Literal["random_weights_in_vendored"] = "random_weights_in_vendored"
    model_class: str  # must expose `from_hf_config_random`
    model_name: str  # HF hub id (config shapes only)


LMTargetSpec = Annotated[
    HFTarget | PretrainedTarget | HFWeightsInVendored | RandomWeightsInVendored,
    Discriminator("kind"),
]


class LMTargetConfig(BaseConfig):
    """Config for the LM target model and how to extract the prediction tensor.

    `output_extract` (passed to `make_run_batch`) pulls the prediction tensor out of the
    model's forward output (default `"logits"`).
    """

    spec: LMTargetSpec
    output_extract: int | str | None = "logits"
    activation_checkpointing: bool = False
    """If True and the target exposes `enable_activation_checkpointing()`, turn on
    per-block gradient checkpointing on the frozen target forward. Trades ~33% extra
    compute for ~10–15x less stored activation memory under 3-pool — the main lever for
    raising `b_per_rank` on deep targets."""
    weights_dtype: Literal["float32", "bfloat16"] = "float32"
    """dtype for the FROZEN target weights. `bfloat16` halves the target's resident footprint
    on every pool (the dominant resident term for an 8B target) — for natively-bf16 models the
    matmuls already run bf16 under autocast, so this only changes residual/norm accumulation
    precision (measured ~5e-4 nats KL on Llama-3.1-8B clean logits, negligible vs recon KLs).
    Only the frozen target is cast; trained V/U components stay fp32 (their AdamW master)."""


class LMDataConfig(BaseConfig):
    """LM experiment dataset / dataloader settings."""

    dataset_name: str = Field(..., description="HuggingFace dataset id")
    data_files: str | None = Field(
        default=None,
        description=(
            "Explicit file glob passed to load_dataset (e.g. 'sample/350BT/*.parquet'). "
            "Resolves directly against that path instead of enumerating the whole repo "
            "tree, which slashes Hub API calls vs. selecting a config by name."
        ),
    )
    revision: str | None = Field(
        default=None,
        description="Dataset git revision (commit SHA/tag) to pin layout and data for reproducibility",
    )
    tokenizer_name: str = Field(..., description="HF tokenizer id or path")
    column_name: str = Field(default="text", description="Dataset column with the text/tokens")
    max_seq_len: PositiveInt = Field(default=512, description="Max sequence length")
    train_split: str = Field(default="train")
    eval_split: str = Field(default="test")
    is_tokenized: bool = Field(default=False)
    streaming: bool = Field(default=False)
    buffer_size: PositiveInt = Field(default=1000)
    synthetic_tokens: bool = Field(
        default=False,
        description=(
            "Benchmarking scaffolding: yield seeded uniform-random token ids instead of "
            "reading the dataset — FLOP-identical batches with zero filesystem traffic. "
            "Requires is_tokenized=True; dataset/streaming fields are ignored."
        ),
    )
    shuffle_each_epoch: bool = Field(default=True)


class LMExperimentConfig(ExperimentConfig[LMTargetConfig, LMDataConfig]):
    pass
