"""Torch-free canonical ↔ concrete path mapping for the app.

The app reads JAX runs natively (no torch model), so it cannot use the torch-coupled
`param_decomp_lab.topology.TransformerTopology` (built from an `nn.Module`). This rebuilds
the same canonical-address mapping from a model-type name and the decomposed site names,
using the shared `_PathSchema` definitions.
"""

from dataclasses import dataclass

from param_decomp_lab.topology.canonical import CanonicalWeight, LayerWeight
from param_decomp_lab.topology.path_schemas import _PathSchema, path_schema_for_model_type


@dataclass(frozen=True)
class AppTopology:
    """Canonical ↔ concrete module-path mapping, built from config (not a live model)."""

    path_schema: _PathSchema
    n_blocks: int

    @classmethod
    def from_model_type(cls, model_type: str, site_names: tuple[str, ...]) -> "AppTopology":
        schema = path_schema_for_model_type(model_type)
        layer_indices = {
            w.layer_idx
            for w in (schema.parse_target_path(s) for s in site_names)
            if isinstance(w, LayerWeight)
        }
        assert layer_indices, f"no per-layer sites among {site_names}"
        return cls(path_schema=schema, n_blocks=max(layer_indices) + 1)

    def canon_to_target(self, canonical: str) -> str:
        return self.path_schema.render_canonical_weight(CanonicalWeight.parse(canonical))

    def target_to_canon(self, target_module_path: str) -> str:
        return self.path_schema.parse_target_path(target_module_path).canonical_str()
