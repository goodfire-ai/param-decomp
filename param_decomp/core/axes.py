"""The two closed axis-name vocabularies of the placement design (PLACEMENT_DESIGN.md).

`MeshAxis` names the logical device-grid axes a run's mesh may declare; `SemanticAxis`
names the tensor dimensions placement rules map onto them. Both are deliberately CLOSED
Literals so a misspelled axis name is a TYPE error, never a silent replication (rule
lookup is exact-name with a quiet unlisted-axis-replicates default). A new consumed
axis is a new Literal member — never a loosening back to `str`. The vocabularies that
stay genuinely open (semantic GROUP names, site names, tap keys) are target-declared
data and remain `str`.

`MeshAxis` is the union over supported mesh SHAPES, not one mesh's axis list: a rule
naming an axis the run's bound mesh does not declare still dies at `PlacedRule`
construction. Dependency-free on purpose — the pydantic config schema (jax-free by
design) and the jax runtime type against the same vocabulary.
"""

from typing import Literal

MeshAxis = Literal["replicate", "fsdp", "tp", "data"]
"""`replicate`/`fsdp`/`tp` — the 3-D HSDP mesh; `data` — the combined data axis of the
two-axis `(data, tp)` mesh."""

SemanticAxis = Literal[
    # component V/U stacks and their faithfulness deltas
    "stack",
    "d_in",
    "d_out",
    "C",
    # the CI transformer's weights; attention keeps DISTINCT query and K/V head axes
    # (GQA: q/o carry n_head, k/v carry n_kv_head — one generic "head" would let a mesh
    # tile one count and silently not the other)
    "d_model",
    "q_head",
    "kv_head",
    "ffn_hidden",
    "input",
    # activation waists (`components.activation_axes`) and the attention head-split view
    "batch",
    "position",
    "feature",
    "head_dim",
    # frozen-target weights
    "layer",
    "vocab",
    "rope_frequency",
]

Axes = tuple[SemanticAxis, ...]

# A semantic dim name -> the mesh axes it shards over (one, an ordered tuple, or None =
# replicated).
MeshAssignment = MeshAxis | tuple[MeshAxis, ...] | None
