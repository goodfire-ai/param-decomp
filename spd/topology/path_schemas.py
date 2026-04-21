"""Path schemas: bidirectional mapping between concrete module paths and canonical weights.

Each model family gets a PathSchema subclass that declares its concrete naming conventions.
These are private implementation details — only TransformerTopology is public.
"""

import re
from abc import ABC
from dataclasses import dataclass
from typing import Literal

from torch import nn

from spd.topology.canonical import (
    CanonicalWeight,
    Embed,
    FusedAttnWeight,
    GLUWeight,
    LayerWeight,
    MLPWeight,
    SeparateAttnWeight,
    Unembed,
)


@dataclass
class _SeparateAttnPathSchema:
    base: str
    q: str
    k: str
    v: str
    o: str

    def _lookup(self) -> dict[str, Literal["q", "k", "v", "o"]]:
        return {self.q: "q", self.k: "k", self.v: "v", self.o: "o"}

    def _reverse(self) -> dict[str, str]:
        return {"q": self.q, "k": self.k, "v": self.v, "o": self.o}

    def parse(self, projection_name: str, layer_idx: int) -> LayerWeight:
        table = self._lookup()
        assert projection_name in table, f"Unknown attn projection: {projection_name}"
        return LayerWeight(layer_idx, SeparateAttnWeight(table[projection_name]))

    def render(self, w: SeparateAttnWeight) -> str:
        return f"{self.base}.{self._reverse()[w.weight]}"


@dataclass
class _FusedAttnPathSchema:
    base: str
    qkv: str
    o: str

    def _lookup(self) -> dict[str, Literal["qkv", "o"]]:
        return {self.qkv: "qkv", self.o: "o"}

    def _reverse(self) -> dict[str, str]:
        return {"qkv": self.qkv, "o": self.o}

    def parse(self, projection_name: str, layer_idx: int) -> LayerWeight:
        table = self._lookup()
        assert projection_name in table, f"Unknown fused attn projection: {projection_name}"
        return LayerWeight(layer_idx, FusedAttnWeight(table[projection_name]))

    def render(self, w: FusedAttnWeight) -> str:
        return f"{self.base}.{self._reverse()[w.weight]}"


@dataclass
class _GLUPathSchema:
    base: str
    gate: str
    up: str
    down: str

    def _lookup(self) -> dict[str, Literal["up", "down", "gate"]]:
        return {self.gate: "gate", self.up: "up", self.down: "down"}

    def _reverse(self) -> dict[str, str]:
        return {"gate": self.gate, "up": self.up, "down": self.down}

    def parse(self, projection_name: str, layer_idx: int) -> LayerWeight:
        table = self._lookup()
        assert projection_name in table, f"Unknown GLU projection: {projection_name}"
        return LayerWeight(layer_idx, GLUWeight(table[projection_name]))

    def render(self, w: GLUWeight) -> str:
        return f"{self.base}.{self._reverse()[w.weight]}"


@dataclass
class _FFNPathSchema:
    base: str
    up: str
    down: str

    def _lookup(self) -> dict[str, Literal["up", "down"]]:
        return {self.up: "up", self.down: "down"}

    def _reverse(self) -> dict[str, str]:
        return {"up": self.up, "down": self.down}

    def parse(self, projection_name: str, layer_idx: int) -> LayerWeight:
        table = self._lookup()
        assert projection_name in table, f"Unknown MLP projection: {projection_name}"
        return LayerWeight(layer_idx, MLPWeight(table[projection_name]))

    def render(self, w: MLPWeight) -> str:
        return f"{self.base}.{self._reverse()[w.weight]}"


class _PathSchema(ABC):
    embedding_path: str
    blocks: str
    attn: _SeparateAttnPathSchema | _FusedAttnPathSchema
    mlp: _GLUPathSchema | _FFNPathSchema
    unembed_path: str
    _block_re: re.Pattern[str] | None = None

    def parse_target_path(self, path: str) -> CanonicalWeight:
        if path == self.embedding_path:
            return Embed()
        if path == self.unembed_path:
            return Unembed()
        return self._parse_block_path(path)

    def render_canonical_weight(self, weight: CanonicalWeight) -> str:
        match weight:
            case Embed():
                return self.embedding_path
            case Unembed():
                return self.unembed_path
            case LayerWeight() as lw:
                return self._render_layer_weight(lw)
            case _:
                raise ValueError(f"Unknown canonical weight: {weight!r}")

    def _parse_block_path(self, path: str) -> LayerWeight:
        """Parse a block-level path like 'h.3.attn.q_proj' into a LayerWeight."""
        if self._block_re is None:
            attn_base = re.escape(self.attn.base)
            mlp_base = re.escape(self.mlp.base)
            blocks = re.escape(self.blocks)
            self._block_re = re.compile(
                rf"^{blocks}\.(?P<idx>\d+)\."
                rf"(?:(?P<attn>{attn_base})\.(?P<attn_proj>\w+)"
                rf"|(?P<mlp>{mlp_base})\.(?P<mlp_proj>\w+))$"
            )

        m = self._block_re.match(path)
        assert m is not None, f"Invalid block path: {path!r}"

        layer_idx = int(m.group("idx"))
        if m.group("attn"):
            return self.attn.parse(m.group("attn_proj"), layer_idx)
        return self.mlp.parse(m.group("mlp_proj"), layer_idx)

    def _render_layer_weight(self, w: LayerWeight) -> str:
        """Render a LayerWeight into a concrete path."""
        base = f"{self.blocks}.{w.layer_idx}"
        match w.name:
            case SeparateAttnWeight() as attn_w:
                assert isinstance(self.attn, _SeparateAttnPathSchema)
                return f"{base}.{self.attn.render(attn_w)}"
            case FusedAttnWeight() as attn_w:
                assert isinstance(self.attn, _FusedAttnPathSchema)
                return f"{base}.{self.attn.render(attn_w)}"
            case GLUWeight() as ffn_w:
                assert isinstance(self.mlp, _GLUPathSchema)
                return f"{base}.{self.mlp.render(ffn_w)}"
            case MLPWeight() as ffn_w:
                assert isinstance(self.mlp, _FFNPathSchema)
                return f"{base}.{self.mlp.render(ffn_w)}"


class _LlamaSimplePathSchema(_PathSchema):
    embedding_path = "wte"
    blocks = "h"
    attn = _SeparateAttnPathSchema(base="attn", q="q_proj", k="k_proj", v="v_proj", o="o_proj")
    mlp = _GLUPathSchema(base="mlp", gate="gate_proj", up="up_proj", down="down_proj")
    unembed_path = "lm_head"


class _LlamaSimpleMLPPathSchema(_PathSchema):
    embedding_path = "wte"
    blocks = "h"
    attn = _SeparateAttnPathSchema(base="attn", q="q_proj", k="k_proj", v="v_proj", o="o_proj")
    mlp = _FFNPathSchema(base="mlp", up="c_fc", down="down_proj")
    unembed_path = "lm_head"


class _GPT2SimplePathSchema(_PathSchema):
    embedding_path = "wte"
    blocks = "h"
    attn = _SeparateAttnPathSchema(base="attn", q="q_proj", k="k_proj", v="v_proj", o="o_proj")
    mlp = _FFNPathSchema(base="mlp", up="c_fc", down="down_proj")
    unembed_path = "lm_head"


class _GPT2PathSchema(_PathSchema):
    embedding_path = "wte"
    blocks = "h_torch"
    attn = _FusedAttnPathSchema(base="attn", qkv="c_attn", o="c_proj")
    mlp = _FFNPathSchema(base="mlp", up="c_fc", down="c_proj")
    unembed_path = "lm_head"


class _HFGpt2PathSchema(_PathSchema):
    embedding_path = "transformer.wte"
    blocks = "transformer.h"
    attn = _FusedAttnPathSchema(base="attn", qkv="c_attn", o="c_proj")
    mlp = _FFNPathSchema(base="mlp", up="c_fc", down="c_proj")
    unembed_path = "lm_head"


def get_path_schema(model: nn.Module) -> _PathSchema:
    from transformers.models.gpt2 import GPT2LMHeadModel

    from spd.pretrain.models import GPT2, GPT2Simple, LlamaSimple, LlamaSimpleMLP

    match model:
        case LlamaSimple():
            return _LlamaSimplePathSchema()
        case LlamaSimpleMLP():
            return _LlamaSimpleMLPPathSchema()
        case GPT2Simple():
            return _GPT2SimplePathSchema()
        case GPT2():
            return _GPT2PathSchema()
        case GPT2LMHeadModel():
            return _HFGpt2PathSchema()
        case _:
            raise ValueError(
                f"Unsupported model class {type(model).__name__}. Add a _PathSchema in path_schemas.py."
            )
