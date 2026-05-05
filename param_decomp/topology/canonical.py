"""Canonical weight types for model-agnostic layer addressing.

Pure data types — no torch dependency. Safe to import anywhere.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, override

_EMBED_RE = re.compile(r"^embed$")
_OUTPUT_RE = re.compile(r"^output$")
_LAYER_RE = re.compile(r"^(?P<layer>\d+)\.(?P<sublayer>attn|attn_fused|glu|mlp)\.(?P<proj>[a-z]+)$")


class CanonicalWeight(ABC):
    @abstractmethod
    def canonical_str(self) -> str: ...

    @staticmethod
    def parse(s: str) -> CanonicalWeight:
        """Parse a canonical address string into a CanonicalWeight."""
        m_embed = _EMBED_RE.match(s)
        m_output = _OUTPUT_RE.match(s)
        m_layer = _LAYER_RE.match(s)

        matches = [m for m in (m_embed, m_output, m_layer) if m is not None]
        assert len(matches) == 1, f"Invalid canonical address: {s!r}"

        if m_embed:
            return Embed()
        if m_output:
            return Unembed()

        assert m_layer is not None
        layer_idx = int(m_layer.group("layer"))
        sublayer = m_layer.group("sublayer")
        proj = m_layer.group("proj")

        cls, valid = _SUBLAYER_PROJECTIONS[sublayer]
        assert proj in valid, f"Invalid projection {proj!r} for {sublayer!r} in {s!r}"
        return LayerWeight(layer_idx, cls(weight=proj))


@dataclass(frozen=True)
class Embed(CanonicalWeight):
    @override
    def canonical_str(self) -> str:
        return "embed"


@dataclass(frozen=True)
class Unembed(CanonicalWeight):
    @override
    def canonical_str(self) -> str:
        return "output"


@dataclass(frozen=True)
class SeparateAttnWeight:
    weight: Literal["q", "k", "v", "o"]


@dataclass(frozen=True)
class FusedAttnWeight:
    weight: Literal["qkv", "o"]


AttnWeight = SeparateAttnWeight | FusedAttnWeight


@dataclass(frozen=True)
class GLUWeight:
    weight: Literal["up", "down", "gate"]


@dataclass(frozen=True)
class MLPWeight:
    weight: Literal["up", "down"]


FFNWeight = GLUWeight | MLPWeight


@dataclass(frozen=True)
class LayerWeight(CanonicalWeight):
    layer_idx: int
    name: AttnWeight | FFNWeight

    @override
    def canonical_str(self) -> str:
        match self.name:
            case SeparateAttnWeight(weight=p):
                return f"{self.layer_idx}.attn.{p}"
            case FusedAttnWeight(weight=p):
                return f"{self.layer_idx}.attn_fused.{p}"
            case GLUWeight(weight=p):
                return f"{self.layer_idx}.glu.{p}"
            case MLPWeight(weight=p):
                return f"{self.layer_idx}.mlp.{p}"


_SUBLAYER_PROJECTIONS: dict[str, tuple[type, tuple[str, ...]]] = {
    "attn": (SeparateAttnWeight, ("q", "k", "v", "o")),
    "attn_fused": (FusedAttnWeight, ("qkv", "o")),
    "glu": (GLUWeight, ("up", "down", "gate")),
    "mlp": (MLPWeight, ("up", "down")),
}
