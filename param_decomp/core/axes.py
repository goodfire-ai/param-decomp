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
    # Axes of the component V/U stacks and their faithfulness deltas. A dense stack
    # uses `stack`, `d_in`, `d_out`, and `C`. A block-factored stack
    # (`components.BlockedFactorization`) adds `expert` — the placement rows' spelling
    # of the block/table axis — and `C_block` for the components within one block, and
    # its `d_in`/`d_out` name the dimensions of one block. `C` stays the flat per-site
    # component axis the mask/CI boundary sees (block-major, of size `n_blocks * C_block`).
    "stack",
    "d_in",
    "d_out",
    "C",
    "expert",
    "C_block",
    # the CI transformer's weights; attention keeps DISTINCT query and K/V head axes
    # (GQA: q/o carry n_head, k/v carry n_kv_head — one generic "head" would let a mesh
    # tile one count and silently not the other)
    "d_model",
    "q_head",
    "kv_head",
    "ffn_hidden",
    "input",
    # independent matrix axes besides `stack`
    "depth",
    "site",
    # the selected CI emission's token-major pick axis (k·c_per_block wide). NOT `C`:
    # its picks are in selection order, carry no block co-location, and replicate at
    # the activation waist rather than sharding over tp.
    "selected_c",
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

MeshAssignment = tuple[MeshAxis, ...]
"""The ordered mesh axes one semantic dim shards over; `()` = replicated. The ONE in-code
spelling of a rule value: the config schema's `str` / `list` / `null` forms are authoring
sugar that the placement parse boundary folds into this tuple, so no consumer ever
branches on a value's shape."""
