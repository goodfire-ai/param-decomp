"""The transformer families' activation-tap vocabulary, shared by the GLU and simple-MLP
targets: `resid.{block}` residual taps + site-input taps (the decomposed site's name,
verbatim). The string forms are the wire format (`Chunk.input_taps` carries them as
pytree-static keys) — everything outside `targets/` treats tap keys as opaque strings and
must not mint or split them.
"""

from collections.abc import Callable
from dataclasses import dataclass

from param_decomp.core.family import ArchFamily

_RESID_PREFIX = "resid."


def resid_tap_key(block: int) -> str:
    """The residual stream ENTERING `block` (before its attention norm). The only mint —
    site-input taps are the site name itself, no mint needed."""
    return f"{_RESID_PREFIX}{block}"


@dataclass(frozen=True)
class _ResidIn:
    block: int


@dataclass(frozen=True)
class _SiteInput:
    """The input activation of decomposed site `name` — the vector its matrix multiplies.
    Width is the site's d_in, NOT the residual width (down/o projections read
    intermediates) — `width_of` consults `d_in_of`."""

    name: str
    block: int


@dataclass(frozen=True)
class TransformerTapGrammar:
    """One resolved target's tap vocabulary, bound to its shape: `n_layer` closes the
    block range, `family` closes the site vocabulary, `d_resid` / `d_in_of(site_name)`
    resolve tap widths. Every query parses fail-closed — a key outside the bound
    vocabulary dies here with its reason, never wraps."""

    family: ArchFamily
    n_layer: int
    d_resid: int
    d_in_of: Callable[[str], int]

    def _parse(self, key: str) -> _ResidIn | _SiteInput:
        if key.startswith(_RESID_PREFIX):
            suffix = key.removeprefix(_RESID_PREFIX)
            assert suffix.isdigit(), (
                f"malformed residual tap {key!r}: expected resid.{{block}} with an integer block"
            )
            block = int(suffix)
            assert 0 <= block < self.n_layer, (
                f"residual tap {key!r} out of range: the target has blocks 0..{self.n_layer - 1}"
            )
            return _ResidIn(block)
        block, _kind = self.family.parse(key)
        assert 0 <= block < self.n_layer, (
            f"site tap {key!r} out of range: the target has blocks 0..{self.n_layer - 1}"
        )
        return _SiteInput(key, block)

    def block_of(self, key: str) -> int:
        """The block a tap reads at: the block a residual tap enters, or the block the
        site lives in."""
        return self._parse(key).block

    def width_of(self, key: str) -> int:
        """Concat width contribution of one tap: `d_resid` for a residual tap, the site's
        d_in for a site-input tap."""
        match self._parse(key):
            case _ResidIn():
                return self.d_resid
            case _SiteInput(name=name):
                return self.d_in_of(name)
