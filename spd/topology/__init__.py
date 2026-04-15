"""Canonical transformer topology.

Two layers:
- canonical.py: Pure data types for model-agnostic layer addressing.
  No torch dependency. Used by serialization, database, frontend layout.
- topology.py: Bidirectional mapping between canonical and concrete module paths.
  Depends on torch.nn and specific model classes.

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

from spd.topology.gradient_connectivity import get_sources_by_target as get_sources_by_target
from spd.topology.topology import TransformerTopology as TransformerTopology
