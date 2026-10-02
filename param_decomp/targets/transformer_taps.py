"""Target-owned one-to-one activation vocabulary shared by transformer targets.

Every key names one physical forward activation. Matrix sites and captured activations are
separate vocabularies: a site names a decomposed weight, while a tap names one array in
the forward. Public capture keys therefore cannot alias the same array.
"""

from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass
from typing import Literal, get_args

from param_decomp.core.family import ArchFamily

_RESID_PREFIX = "resid."
_SITE_OUTPUT_SUFFIX = ".out"

SiteInputTapName = Literal["attn_in", "attn_out", "mlp_in", "mlp_hidden"]
"""The per-block vectors a transformer's matrix sites read: the normalized residual the
attention input projections consume, the attention core the o projection consumes, the
normalized post-attention residual the MLP input projections consume, and the
post-nonlinearity MLP activation the down projection consumes."""

BlockTapName = Literal["attn_in", "attn_out", "post_attn", "mlp_in", "mlp_hidden"]
"""Every named per-block vector, in forward order: the site inputs plus `post_attn`, the
raw residual after the block's attention add and before its MLP norm. A target declares
the subset it computes as dense per-block arrays."""

BLOCK_TAP_NAMES: tuple[BlockTapName, ...] = get_args(BlockTapName)
SITE_INPUT_TAP_NAMES: tuple[SiteInputTapName, ...] = get_args(SiteInputTapName)
assert set(SITE_INPUT_TAP_NAMES) < set(BLOCK_TAP_NAMES), (SITE_INPUT_TAP_NAMES, BLOCK_TAP_NAMES)


def block_tap_key(name: BlockTapName, block: int) -> str:
    return f"{name}.{block}"


def resid_tap_key(boundary: int) -> str:
    """Raw residual boundary: 0 enters the first block; ``n_layer`` exits the last."""
    return f"{_RESID_PREFIX}{boundary}"


def post_attention_tap_key(block: int) -> str:
    return block_tap_key("post_attn", block)


def attention_input_tap_key(block: int) -> str:
    return block_tap_key("attn_in", block)


def attention_output_tap_key(block: int) -> str:
    return block_tap_key("attn_out", block)


def mlp_input_tap_key(block: int) -> str:
    return block_tap_key("mlp_in", block)


def mlp_hidden_tap_key(block: int) -> str:
    return block_tap_key("mlp_hidden", block)


def site_output_tap_key(site: str) -> str:
    """Linear output of ``site``, before a following bias, nonlinearity, or residual add."""
    return f"{site}{_SITE_OUTPUT_SUFFIX}"


@dataclass(frozen=True, kw_only=True)
class ResidualBoundary:
    boundary: int


@dataclass(frozen=True, kw_only=True)
class BlockTap:
    name: BlockTapName
    block: int


@dataclass(frozen=True, kw_only=True)
class SiteOutput:
    name: str
    block: int
    kind: str


TransformerPoint = ResidualBoundary | BlockTap | SiteOutput


def _parse_numbered_key(key: str, prefix: str, upper: int, noun_for_err: str) -> int:
    suffix = key.removeprefix(prefix)
    assert suffix.isdigit(), f"malformed {noun_for_err} {key!r}: expected {prefix}{{integer}}"
    index = int(suffix)
    assert 0 <= index <= upper, f"{noun_for_err} {key!r} out of range: expected 0..{upper}"
    return index


@dataclass(frozen=True, kw_only=True)
class BlockCaptures:
    """What one block of a target computes as capturable arrays, with feature widths."""

    tap_widths: Mapping[BlockTapName, int]
    site_output_widths: Mapping[str, int]
    """Matrix kind -> output width, for exactly the kinds this block carries."""


@dataclass(frozen=True, kw_only=True)
class TransformerTapGrammar:
    """One transformer target's closed, one-to-one point grammar.

    ``family`` owns matrix-output syntax; ``blocks`` is the target's own declaration of
    what each block computes. Every query parses fail-closed; core never imports or
    interprets this type.
    """

    family: ArchFamily
    d_resid: int
    blocks: tuple[BlockCaptures, ...]

    @property
    def n_layer(self) -> int:
        return len(self.blocks)

    def site_input_tap_keys(self, blocks: tuple[int, ...]) -> tuple[str, ...]:
        """Every site-input vector the target computes in the requested blocks, once each."""
        assert len(set(blocks)) == len(blocks), blocks
        assert all(0 <= block < self.n_layer for block in blocks), (blocks, self.n_layer)
        return tuple(
            block_tap_key(name, block)
            for block in blocks
            for name in SITE_INPUT_TAP_NAMES
            if name in self.blocks[block].tap_widths
        )

    def parse(self, key: str) -> TransformerPoint:
        if key.startswith(_RESID_PREFIX):
            return ResidualBoundary(
                boundary=_parse_numbered_key(key, _RESID_PREFIX, self.n_layer, "residual boundary")
            )
        if key.endswith(_SITE_OUTPUT_SUFFIX):
            name = key.removesuffix(_SITE_OUTPUT_SUFFIX)
            block, kind = self.family.parse(name)
            assert 0 <= block < self.n_layer, (
                f"site point {key!r} out of range: target blocks are 0..{self.n_layer - 1}"
            )
            assert kind in self.blocks[block].site_output_widths, (
                f"{key!r}: block {block} carries no {kind!r} matrix"
            )
            return SiteOutput(name=name, block=block, kind=kind)
        for name in BLOCK_TAP_NAMES:
            prefix = f"{name}."
            if key.startswith(prefix):
                block = _parse_numbered_key(key, prefix, self.n_layer - 1, "block tap")
                assert name in self.blocks[block].tap_widths, (
                    f"{key!r}: this target computes no {name!r} vector in block {block}"
                )
                return BlockTap(name=name, block=block)
        raise AssertionError(f"unknown transformer activation {key!r}")

    def resolve[SourceT: Hashable](
        self, keys: tuple[str, ...], source_of: Callable[[TransformerPoint], SourceT]
    ) -> tuple[SourceT, ...]:
        """Validate ``keys`` and return their physical sources in request order."""
        sources = tuple(source_of(self.parse(key)) for key in keys)
        assert len(set(sources)) == len(sources), (
            "multiple capture keys name one physical activation",
            keys,
            sources,
        )
        return sources

    def block_of(self, key: str) -> int:
        match self.parse(key):
            case ResidualBoundary(boundary=boundary):
                return boundary
            case BlockTap(block=block) | SiteOutput(block=block):
                return block

    def width_of(self, key: str) -> int:
        match self.parse(key):
            case ResidualBoundary():
                return self.d_resid
            case BlockTap(name=name, block=block):
                return self.blocks[block].tap_widths[name]
            case SiteOutput(block=block, kind=kind):
                return self.blocks[block].site_output_widths[kind]
