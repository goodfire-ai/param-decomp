"""Shared prompt-building helpers for autointerp and graph interpretation.

Pure functions for formatting component data into LLM prompt sections.
"""

import re

from spd.app.backend.app_tokenizer import AppTokenizer
from spd.app.backend.utils import delimit_tokens
from spd.harvest.analysis import TokenPRLift
from spd.harvest.schemas import ComponentData

DATASET_DESCRIPTIONS: dict[str, str] = {
    "SimpleStories/SimpleStories": (
        "SimpleStories: 2M+ short stories (200-350 words), grade 1-8 reading level. "
        "Simple vocabulary, common narrative elements."
    ),
    "danbraunai/pile-uncopyrighted-tok-shuffled": (
        "The Pile (uncopyrighted subset): diverse text from books, "
        "academic papers, code, web pages, and other sources."
    ),
}

WEIGHT_NAMES: dict[str, str] = {
    "attn.q": "attention query projection",
    "attn.k": "attention key projection",
    "attn.v": "attention value projection",
    "attn.o": "attention output projection",
    "mlp.up": "MLP up-projection",
    "mlp.down": "MLP down-projection",
    "glu.up": "GLU up-projection",
    "glu.down": "GLU down-projection",
    "glu.gate": "GLU gate projection",
}

_ORDINALS = ["1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th"]


def ordinal(n: int) -> str:
    if 1 <= n <= len(_ORDINALS):
        return _ORDINALS[n - 1]
    return f"{n}th"


def human_layer_desc(canonical: str, n_blocks: int) -> str:
    """Convert canonical layer string to human-readable description.

    '0.mlp.up' -> 'MLP up-projection in the 1st of 4 blocks'
    '1.attn.q' -> 'attention query projection in the 2nd of 4 blocks'
    """
    m = re.match(r"(\d+)\.(.*)", canonical)
    if not m:
        return canonical
    layer_idx = int(m.group(1))
    weight_key = m.group(2)
    weight_name = WEIGHT_NAMES.get(weight_key, weight_key)
    return f"{weight_name} in the {ordinal(layer_idx + 1)} of {n_blocks} blocks"


def layer_position_note(canonical: str, n_blocks: int) -> str:
    """Brief note about what layer position means for interpretation."""
    m = re.match(r"(\d+)\.", canonical)
    if not m:
        return ""
    layer_idx = int(m.group(1))
    if layer_idx == n_blocks - 1:
        return "This is in the final block, so its output directly influences token predictions."
    remaining = n_blocks - 1 - layer_idx
    return (
        f"This is {remaining} block{'s' if remaining > 1 else ''} from the output, "
        f"so its effect on token predictions is indirect — filtered through later layers."
    )


def density_note(firing_density: float) -> str:
    if firing_density > 0.15:
        return (
            "This is a high-density component (fires frequently). "
            "High-density components often act as broad biases rather than selective features."
        )
    if firing_density < 0.005:
        return "This is a very sparse component, likely highly specific."
    return ""


def build_output_section(
    output_stats: TokenPRLift,
    output_pmi: list[tuple[str, float]] | None,
) -> str:
    section = ""

    if output_pmi:
        section += (
            "**Output PMI (pointwise mutual information, in nats: how much more likely "
            "a token is to be produced when this component fires, vs its base rate. "
            "0 = no association, 1 = ~3x more likely, 2 = ~7x, 3 = ~20x):**\n"
        )
        for tok, pmi in output_pmi[:10]:
            section += f"- {repr(tok)}: {pmi:.2f}\n"

    if output_stats.top_precision:
        section += "\n**Output precision — of all probability mass for token X, what fraction is at positions where this component fires?**\n"
        for tok, prec in output_stats.top_precision[:10]:
            section += f"- {repr(tok)}: {prec * 100:.0f}%\n"

    return section


def build_input_section(
    input_stats: TokenPRLift,
    input_pmi: list[tuple[str, float]] | None,
) -> str:
    section = ""

    if input_pmi:
        section += "**Input PMI (same metric as above, for input tokens):**\n"
        for tok, pmi in input_pmi[:6]:
            section += f"- {repr(tok)}: {pmi:.2f}\n"

    if input_stats.top_precision:
        section += "\n**Input precision — probability the component fires given the current token is X:**\n"
        for tok, prec in input_stats.top_precision[:8]:
            section += f"- {repr(tok)}: {prec * 100:.0f}%\n"

    return section


def build_fires_on_examples(
    component: ComponentData,
    app_tok: AppTokenizer,
    max_examples: int,
) -> str:
    section = ""
    examples = component.activation_examples[:max_examples]

    for i, ex in enumerate(examples):
        if any(ex.firings):
            spans = app_tok.get_spans(ex.token_ids)
            tokens = list(zip(spans, ex.firings, strict=True))
            section += f"{i + 1}. {delimit_tokens(tokens)}\n"

    return section


def build_says_examples(
    component: ComponentData,
    app_tok: AppTokenizer,
    max_examples: int,
) -> str:
    section = ""
    examples = component.activation_examples[:max_examples]

    for i, ex in enumerate(examples):
        if any(ex.firings):
            spans = app_tok.get_spans(ex.token_ids)
            shifted_firings = [False] + ex.firings[:-1]
            tokens = list(zip(spans, shifted_firings, strict=True))
            section += f"{i + 1}. {delimit_tokens(tokens)}\n"

    return section
