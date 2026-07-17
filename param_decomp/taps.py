"""The activation-tap address grammar: typed addresses for the activations a target can
serve through `read_activations`.

One vocabulary, owned here, with `tap_key`/`parse_tap` as the two directions — consumers
(the CI-fn chunk resolver lab-side, the targets' `read_activations`, the hidden-acts eval)
must not mint or split key strings themselves. The string forms are the wire format
(`Chunk.input_taps` carries them as pytree-static keys); the types are the grammar.

Related, deliberately separate for now: `param_decomp_lab/topology/` (the consumer-side
canonical weight/component address space) and #966's `site_tree.ArchFamily` (the site-name
grammar). All three name locations in the same transformer; the intended end state is one
core grammar (this module folds into the site tree when the #966 carve ports) — tracked on
task:review-and-port-pr-966-to-param-decomp-dev.
"""

from dataclasses import dataclass

_RESID_PREFIX = "resid."


@dataclass(frozen=True)
class ResidIn:
    """The residual stream ENTERING block `layer` (before its attention norm)."""

    layer: int


@dataclass(frozen=True)
class SiteInput:
    """The input activation of decomposed site `name` — the vector its matrix multiplies.

    Width is the site's `d_in`, NOT the residual width (down/o projections read
    intermediates), so consumers deriving concat widths must consult the site dims.
    """

    name: str


TapAddress = ResidIn | SiteInput


def tap_key(tap: TapAddress) -> str:
    """The canonical string form — the only place these strings are minted."""
    match tap:
        case ResidIn(layer):
            return f"{_RESID_PREFIX}{layer}"
        case SiteInput(name):
            return name


def parse_tap(key: str) -> TapAddress:
    """Inverse of `tap_key`. Anything that isn't a `resid.{L}` form is a site name; the
    caller validates site names against its own site grammar (they are target-specific)."""
    if key.startswith(_RESID_PREFIX):
        return ResidIn(int(key.removeprefix(_RESID_PREFIX)))
    return SiteInput(key)
