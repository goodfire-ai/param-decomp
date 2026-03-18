"""Shared prompt-building helpers for autointerp and graph interpretation.

Pure functions for formatting component data into LLM prompt sections.
"""

import re

from spd.app.backend.app_tokenizer import AppTokenizer
from spd.app.backend.utils import delimit_tokens
from spd.autointerp.schemas import DECOMPOSITION_DESCRIPTIONS, DecompositionMethod
from spd.harvest.analysis import TokenPRLift
from spd.harvest.schemas import ComponentData
from spd.utils.markdown import Md

DATASET_DESCRIPTIONS: dict[str, str] = {
    "SimpleStories/SimpleStories": (
        "SimpleStories: 2M+ short stories (200-350 words), grade 1-8 reading level. "
        "Simple vocabulary, common narrative elements."
    ),
    "danbraunai/pile-uncopyrighted-tok-shuffled": (
        "The Pile (uncopyrighted subset): diverse text from books, "
        "academic papers, code, web pages, and other sources."
    ),
    "danbraunai/pile-uncopyrighted-tok": (
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
    """'0.mlp.up' -> 'MLP up-projection in the 1st of 4 blocks'"""
    m = re.match(r"(\d+)\.(.*)", canonical)
    if not m:
        return canonical
    layer_idx = int(m.group(1))
    weight_key = m.group(2)
    weight_name = WEIGHT_NAMES.get(weight_key, weight_key)
    return f"{weight_name} in the {ordinal(layer_idx + 1)} of {n_blocks} blocks"


def layer_position_note(canonical: str, n_blocks: int) -> str:
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


def build_data_presentation(
    seq_len: int,
    context_tokens_per_side: int,
    decomposition_method: DecompositionMethod,
) -> Md:
    window_size = 2 * context_tokens_per_side + 1
    md = Md()

    md.h(3, "Decomposition method")
    md.p(DECOMPOSITION_DESCRIPTIONS[decomposition_method])

    md.h(3, "Data")
    md.p(
        f"The model processes sequences of {seq_len} tokens. "
        f"Each activation example below shows a {window_size}-token window centered on the "
        f"firing token, with up to {context_tokens_per_side} tokens of context on each side. "
        f"Windows are truncated at sequence boundaries. "
        f"Examples are sampled uniformly at random from all firings across the dataset."
    )

    md.h(3, "Metric definitions")
    md.p("The token statistics below use these metrics:")
    md.bullets(
        [
            "**Precision**: P(component fires | token). Of all occurrences of token X in "
            "the dataset, what fraction had this component firing?",
            "**PMI** (pointwise mutual information, in nats): How much more likely is "
            "co-occurrence than chance? 0 = no association, 1 ≈ 3x, 2 ≈ 7x, 3 ≈ 20x.",
        ]
    )
    md.p(
        "**Input** metrics concern the token at the position where the component fires. "
        "**Output** metrics concern what the model predicts (at its final logits) at "
        "those positions — not the component's direct output."
    )
    return md


def build_output_section(
    output_stats: TokenPRLift,
    output_pmi: list[tuple[str, float]] | None,
) -> Md:
    md = Md()
    if output_pmi:
        md.labeled_list(
            "**Output PMI:**",
            [f"{repr(tok)}: {pmi:.2f}" for tok, pmi in output_pmi[:10]],
        )
    if output_stats.top_precision:
        md.labeled_list(
            "**Output precision:**",
            [f"{repr(tok)}: {prec * 100:.0f}%" for tok, prec in output_stats.top_precision[:10]],
        )
    return md


def build_input_section(
    input_stats: TokenPRLift,
    input_pmi: list[tuple[str, float]] | None,
) -> Md:
    md = Md()
    if input_pmi:
        md.labeled_list(
            "**Input PMI:**",
            [f"{repr(tok)}: {pmi:.2f}" for tok, pmi in input_pmi[:6]],
        )
    if input_stats.top_precision:
        md.labeled_list(
            "**Input precision:**",
            [f"{repr(tok)}: {prec * 100:.0f}%" for tok, prec in input_stats.top_precision[:8]],
        )
    return md


def _build_examples(
    component: ComponentData,
    app_tok: AppTokenizer,
    max_examples: int,
    shift_firings: bool,
) -> Md:
    items: list[str] = []
    for ex in component.activation_examples[:max_examples]:
        if not any(ex.firings):
            continue
        spans = app_tok.get_spans(ex.token_ids)
        firings = [False] + ex.firings[:-1] if shift_firings else ex.firings
        tokens = list(zip(spans, firings, strict=True))
        items.append(delimit_tokens(tokens))
    md = Md()
    if items:
        md.numbered(items)
    return md


def build_fires_on_examples(
    component: ComponentData,
    app_tok: AppTokenizer,
    max_examples: int,
) -> Md:
    return _build_examples(component, app_tok, max_examples, shift_firings=False)


def build_says_examples(
    component: ComponentData,
    app_tok: AppTokenizer,
    max_examples: int,
) -> Md:
    return _build_examples(component, app_tok, max_examples, shift_firings=True)


def _fmt_ann(activations: dict[str, float]) -> str:
    """Format activation annotations for a single firing token.

    For SPD: (ci:0.82, act:-0.05)
    For other methods: just the activation value, e.g. (act:3.21)
    """
    parts: list[str] = []
    if "causal_importance" in activations:
        parts.append(f"ci:{activations['causal_importance']:.2g}")
    if "component_activation" in activations:
        parts.append(f"act:{activations['component_activation']:.2g}")
    if "activation" in activations:
        parts.append(f"act:{activations['activation']:.2g}")
    return f"({', '.join(parts)})"


def _delimit_annotated(
    spans: list[str],
    firings: list[bool],
    per_token_activations: list[dict[str, float]],
) -> str:
    """Join token strings, wrapping active tokens with <<<token (ci, act)>>>."""
    parts: list[str] = []
    for span, active, acts in zip(spans, firings, per_token_activations, strict=True):
        if active:
            stripped = span.lstrip()
            whitespace = span[: len(span) - len(stripped)]
            ann = _fmt_ann(acts)
            parts.append(f"{whitespace}<<<{stripped} {ann}>>>")
        else:
            parts.append(span)
    return "".join(parts)


def build_annotated_examples(
    component: ComponentData,
    app_tok: AppTokenizer,
    max_examples: int,
) -> Md:
    """Build activation examples with per-token CI and activation annotations."""
    items: list[str] = []
    for ex in component.activation_examples[:max_examples]:
        if not any(ex.firings):
            continue
        spans = app_tok.get_spans(ex.token_ids)
        act_keys = list(ex.activations.keys())
        per_token_acts = [
            {k: ex.activations[k][i] for k in act_keys} for i in range(len(ex.token_ids))
        ]
        items.append(_delimit_annotated(spans, ex.firings, per_token_acts))
    md = Md()
    if items:
        md.numbered(items)
    return md
