"""Source masks carry one CI coordinate frame and the source payload aligned to it."""

from dataclasses import dataclass

import jax
from jaxtyping import Array

from param_decomp.core.components import SiteCI, map_site_ci


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SourceMaskIngredients:
    """CI and source values in one frame, read by ``read_source_mask`` and stacked as one product.

    Selected routing belongs only to ``ci``. The source payload has already been read
    in that order; composition neither accepts nor discards a second routing frame.
    Delta values retain their independent logical token coordinates.
    """

    ci: SiteCI
    source_values: Array
    delta: Array

    def compose(self) -> SiteCI:
        """Interpolate CI and source values in their shared component coordinates."""
        return map_site_ci(lambda ci: ci + (1.0 - ci) * self.source_values, self.ci)
