"""Render the per-point run yaml (the ``RunConfig`` consumed by the launcher).

Currently uses an f-string template — concise and readable, but bypasses
:class:`LMRunConfig`'s schema. ``validate_run_yaml`` round-trips the rendered
text through ``RunConfig.from_dict`` to catch schema drift at submit time
(rather than at SLURM-execution time, when an OOM-cliff-priced cluster slot
is already burning).
"""

from typing import Any

import yaml

from param_decomp.scripts.two_pool_benchmark.submit_sweep.schema import ModelSpec, SweepPoint


def render_run_yaml(
    *,
    name: str,
    model: ModelSpec,
    point: SweepPoint,
    topology_label: str,
) -> str:
    """Render the run.yaml string for one sweep point.

    Note: the f-string approach is brittle in principle (schema drift goes
    undetected at render time). :func:`validate_run_yaml` round-trips the
    result through :class:`RunConfig.from_dict` to enforce the schema.
    """
    ci = point.ci.resolved()
    return f"""driver_path: param_decomp.experiments.lm.experiment:Driver
name: {name}
view_meta:
  batch: {point.batch}
  seq: {point.seq}
  ci_d: {ci.d}
  ci_n: {ci.n_blocks}
  topology: {topology_label}
pd:
  seed: {point.seed}
  n_mask_samples: 1
  ci_config:
    mode: layerwise
    fn_type: transformer
    hidden_dims: null
    transformer_cfg:
      d_model: {ci.d}
      n_blocks: {ci.n_blocks}
      mlp_hidden_dim:
      - {ci.mlp_hidden}
      attn_config:
        n_heads: {ci.n_heads}
        max_len: {point.seq}
        rope_base: 10000.0
  sampling: continuous
  sigmoid_type: leaky_hard
  module_info:
  - module_pattern: model.layers.*.self_attn.q_proj
    C: 32
  - module_pattern: model.layers.*.self_attn.k_proj
    C: 32
  - module_pattern: model.layers.*.self_attn.v_proj
    C: 32
  - module_pattern: model.layers.*.self_attn.o_proj
    C: 32
  - module_pattern: model.layers.*.mlp.gate_proj
    C: 32
  - module_pattern: model.layers.*.mlp.up_proj
    C: 32
  - module_pattern: model.layers.*.mlp.down_proj
    C: 32
  identity_module_info: null
  use_delta_component: true
  components_optimizer:
    lr_schedule:
      start_val: 5.0e-05
      warmup_pct: 0.0
      final_val_frac: 1.0
      fn_type: constant
    grad_clip_norm: null
  ci_fn_optimizer:
    lr_schedule:
      start_val: 5.0e-05
      warmup_pct: 0.0
      final_val_frac: 1.0
      fn_type: constant
  steps: {point.steps}
  batch_size: {point.batch}
  faithfulness_warmup_steps: 0
  faithfulness_warmup_lr: 0.001
  faithfulness_warmup_weight_decay: 0.0
  loss_metrics:
    FaithfulnessLoss:
      coeff: 1.0e+06
    ImportanceMinimalityLoss:
      coeff: 1.0e-04
      pnorm: 1.0
      beta: 0.5
      p_anneal_start_frac: 0.0
      p_anneal_final_p: 1.0
      p_anneal_end_frac: 1.0
      eps: 1.0e-12
    StochasticReconLayerwiseLoss:
      coeff: 0.5
    PersistentPGDReconLoss:
      coeff: 0.5
      optimizer:
        type: sign
        lr_schedule:
          start_val: 0.01
          warmup_pct: 0.0
          final_val_frac: 1.0
          fn_type: constant
      scope:
        type: per_batch_per_position
      use_sigmoid_parameterization: false
      n_warmup_steps: 3
      n_samples: 1
logging:
  train_log_freq: 10
  eval_freq: 100000
  slow_eval_freq: 100000
  slow_eval_on_first_step: false
  n_eval_steps: 1
  eval_batch_size: {point.batch}
  save_freq: null
  eval_metrics: {{}}
runtime:
  autocast_bf16: true
  device: cuda
  dp: null
target:
  model_class: transformers.AutoModelForCausalLM
  model_name: {model.name}
  model_path: null
  output_extract: logits
data:
  tokenizer_name: {model.name}
  max_seq_len: {point.seq}
  buffer_size: 1000
  dataset_name: random
  column_name: input_ids
  train_split: train
  eval_split: train
  shuffle_each_epoch: false
  is_tokenized: true
  streaming: false
  is_random: true
  random_vocab_size: {model.vocab_size}
"""


def validate_run_yaml(text: str) -> None:
    """Round-trip the rendered yaml through ``RunConfig.from_dict``.

    Raises if the schema is violated. Catches drift between this module's
    f-string and ``LMRunConfig`` at submit time instead of SLURM run time.

    Imports are deferred so the CLI's import latency doesn't pay for torch +
    transformers unless we're about to actually submit.
    """
    from param_decomp.run import RunConfig

    parsed: Any = yaml.safe_load(text)
    RunConfig.from_dict(parsed)  # raises on schema mismatch
