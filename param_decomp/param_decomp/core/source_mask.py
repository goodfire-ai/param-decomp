"""Source masks carry one CI coordinate frame and the source payload aligned to it."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from param_decomp.core.components import SiteCI, map_site_ci, site_ci_values
from param_decomp.core.linear_plan import uniform_like


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class DocumentMixedSource:
    """Each document's source: `persistent` where `adversarial`, else fresh uniform noise
    from `key`. The noise is drawn where the mask is composed, so a target composing
    inside its checkpointed blocks holds at most one block's noise, and its backward
    redraws it from the key. `adversarial` holds one flag per document."""

    adversarial: Array
    persistent: Array
    key: PRNGKeyArray

    def values(self, reference: Array, *, drop_last_axis: bool) -> Array:
        """The mixed source over `reference`'s coordinates (less its last axis when
        `drop_last_axis`) in `reference`'s dtype; the noise is drawn in fp32, then cast."""
        noise = uniform_like(
            self.key, reference, drop_last_axis=drop_last_axis, dtype=jnp.float32
        ).astype(reference.dtype)
        selector = self.adversarial.reshape(
            self.adversarial.shape[0], *(1 for _ in noise.shape[1:])
        ).astype(self.persistent.dtype)
        return selector * self.persistent + (1 - selector) * noise


type SourcePayload = Array | DocumentMixedSource
"""Source values already read into the CI frame, or a per-document mix with fresh noise."""


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SourceMaskIngredients:
    """CI and source values in one frame, read by ``read_source_mask`` and stacked as one product.

    Selected routing belongs only to ``ci``. The source payload has already been read
    in that order; composition neither accepts nor discards a second routing frame.
    Delta values retain their independent logical token coordinates. A payload is
    either read values or a per-document mix with fresh noise, drawn at composition.
    """

    ci: SiteCI
    source_values: SourcePayload
    delta: SourcePayload

    def compose(self) -> SiteCI:
        """Interpolate CI and source values in their shared component coordinates."""
        values = self._values(self.source_values, drop_last_axis=False)
        return map_site_ci(lambda ci: ci + (1.0 - ci) * values, self.ci)

    def delta_mask(self) -> Array:
        """The weight-delta mask: one value per token."""
        return self._values(self.delta, drop_last_axis=True)

    def _values(self, payload: SourcePayload, *, drop_last_axis: bool) -> Array:
        match payload:
            case DocumentMixedSource():
                return payload.values(site_ci_values(self.ci), drop_last_axis=drop_last_axis)
            case jax.Array():
                return payload
