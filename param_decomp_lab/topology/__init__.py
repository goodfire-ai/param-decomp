"""Canonical transformer topology.

Two layers:
- canonical.py: Pure data types for model-agnostic layer addressing.
- path_schemas.py: Bidirectional mapping between canonical and concrete module paths,
  selected by model-type name (`path_schema_for_model_type`) — torch-free, no live model.

Canonical layer address format:
    "embed"                   — embedding
    "output"                  — unembed / logits
    "{block}.attn.{p}"        — separate attention (p: q | k | v | o)
    "{block}.attn_fused.{p}"  — fused attention (p: qkv | o)
    "{block}.glu.{p}"         — gated FFN / SwiGLU (p: up | down | gate)
    "{block}.mlp.{p}"         — simple FFN (p: up | down)

Node key format:
    "{layer_address}:{seq_pos}:{component_idx}"
"""

from param_decomp_lab.topology.path_schemas import (
    path_schema_for_model_type as path_schema_for_model_type,
)
