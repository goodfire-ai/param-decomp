"""Shared prompt-building helpers for autointerp and graph interpretation.

Pure functions for formatting component data into LLM prompt sections.
"""

import re
from typing import Literal

from spd.app.backend.app_tokenizer import AppTokenizer
from spd.app.backend.utils import delimit_tokens
from spd.autointerp.config import (
    ExampleRenderingConfig,
    LegacyDelimitedExamplesConfig,
    SingleLineExamplesConfig,
    XmlExamplesConfig,
)
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


def token_stats_section(
    stats: TokenPRLift,
    direction: Literal["Input", "Output"],
    direction_description: str,
) -> Md:
    md = Md()
    md.h(3, f"{direction} tokens ({direction_description})")
    if stats.top_recall:
        md.labeled_list(
            f"**Most common {direction.lower()} tokens** (when the component fires, what "
            f"fraction of {'the model\u2019s next-token probability mass goes to' if direction == 'Output' else 'the time the current token is'} "
            "token X):",
            [f"`{repr(tok)}`: {recall * 100:.0f}%" for tok, recall in stats.top_recall[:8]],
        )
    if stats.top_pmi:
        md.labeled_list(
            f"**{direction} PMI** (same data normalized by base rate \u2014 how much more "
            "likely than usual; nats: 0 = none, 1 \u2248 3\u00d7, 2 \u2248 7\u00d7, "
            "3 \u2248 20\u00d7):",
            [f"`{repr(tok)}`: {pmi:.2f}" for tok, pmi in stats.top_pmi[:10]],
        )
    return md


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


def _highlight_example(rendering: SingleLineExamplesConfig | XmlExamplesConfig) -> str:
    match rendering:
        case SingleLineExamplesConfig(highlight_delimiter="angle", annotation_style="none"):
            return "`<<<token>>>`"
        case SingleLineExamplesConfig(highlight_delimiter="angle", annotation_style="activation"):
            return "`<<<token (ci:X, act:Y)>>>`"
        case SingleLineExamplesConfig(highlight_delimiter="brackets", annotation_style="none"):
            return "`[[[token]]]`"
        case SingleLineExamplesConfig(
            highlight_delimiter="brackets", annotation_style="activation"
        ):
            return "`[[[token]]] (ci:X, act:Y)`"
        case XmlExamplesConfig(highlight_delimiter="angle", annotation_style="none"):
            return "`<<<token>>>`"
        case XmlExamplesConfig(highlight_delimiter="angle", annotation_style="activation"):
            return "`<<<token (ci:X, act:Y)>>>`"
        case XmlExamplesConfig(highlight_delimiter="brackets", annotation_style="none"):
            return "`[[[token]]]`"
        case XmlExamplesConfig(highlight_delimiter="brackets", annotation_style="activation"):
            return "`[[[token (ci:X, act:Y)]]]`"
        case _:
            raise AssertionError(f"Unhandled rendering: {rendering}")


def describe_example_rendering(rendering: ExampleRenderingConfig) -> str:
    match rendering:
        case LegacyDelimitedExamplesConfig():
            return (
                "Examples use grouped `<<delimiters>>` around contiguous active tokens. "
                "Tokens inside the delimiters are positions where the component is active."
            )
        case SingleLineExamplesConfig(annotation_style="activation") as single_line:
            return (
                "Each example is one annotated line. Firing tokens are wrapped as "
                f"{_highlight_example(single_line)}. `ci` is causal importance and `act` "
                "is the component activation at that position. Control characters are "
                "rendered visibly, e.g. newline as `↵`."
            )
        case SingleLineExamplesConfig() as single_line:
            return (
                "Each example is one line with firing tokens wrapped as "
                f"{_highlight_example(single_line)}. Control characters are rendered "
                "visibly, e.g. newline as `↵`."
            )
        case XmlExamplesConfig() as xml:
            raw_desc = (
                "The `<raw>` block sanitizes control characters for readability, e.g. newline as `↵`."
                if xml.sanitize_raw
                else "The `<raw>` block preserves literal whitespace and control characters."
            )
            highlighted_desc = (
                "The `<highlighted>` block sanitizes control characters for readability while preserving token boundaries."
                if xml.sanitize_highlighted
                else "The `<highlighted>` block also preserves literal token text."
            )
            return (
                "Each example is an XML-style block with `<raw>` and `<highlighted>` sections. "
                f"`<highlighted>` repeats the same window with firing tokens wrapped as "
                f"{_highlight_example(xml)}. {raw_desc} {highlighted_desc}"
            )


def _build_legacy_examples(
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


def _split_display_span(
    display_span: str,
    token_id: int,
    app_tok: AppTokenizer,
    sanitize: bool,
) -> tuple[str, str]:
    stripped = display_span.lstrip()
    whitespace = display_span[: len(display_span) - len(stripped)]
    if stripped:
        return whitespace, stripped
    token_text = app_tok.get_tok_display(token_id) if sanitize else app_tok.decode([token_id])
    return whitespace, token_text


def _delimiter_pair(style: Literal["brackets", "angle"]) -> tuple[str, str]:
    if style == "brackets":
        return ("[[[", "]]]")
    return ("<<<", ">>>")


def _single_line_token(
    display_span: str,
    token_id: int,
    ann: str | None,
    app_tok: AppTokenizer,
    delimiter_style: Literal["brackets", "angle"],
) -> str:
    whitespace, token_text = _split_display_span(display_span, token_id, app_tok, sanitize=True)
    open_delim, close_delim = _delimiter_pair(delimiter_style)
    if not ann:
        return f"{whitespace}{open_delim}{token_text}{close_delim}"
    if delimiter_style == "angle":
        return f"{whitespace}{open_delim}{token_text} {ann}{close_delim}"
    return f"{whitespace}{open_delim}{token_text}{close_delim} {ann}"


def _xml_token(
    display_span: str,
    token_id: int,
    ann: str | None,
    app_tok: AppTokenizer,
    delimiter_style: Literal["brackets", "angle"],
    sanitize: bool,
) -> str:
    whitespace, token_text = _split_display_span(display_span, token_id, app_tok, sanitize=sanitize)
    open_delim, close_delim = _delimiter_pair(delimiter_style)
    if not ann:
        return f"{whitespace}{open_delim}{token_text}{close_delim}"
    return f"{whitespace}{open_delim}{token_text} {ann}{close_delim}"


def _build_single_line_text(
    spans: list[str],
    token_ids: list[int],
    firings: list[bool],
    per_token_activations: list[dict[str, float]],
    app_tok: AppTokenizer,
    rendering: SingleLineExamplesConfig,
) -> str:
    parts: list[str] = []
    for token_id, span, active, acts in zip(
        token_ids, spans, firings, per_token_activations, strict=True
    ):
        if not active:
            parts.append(span)
            continue
        ann = _fmt_ann(acts) if rendering.annotation_style == "activation" else None
        parts.append(
            _single_line_token(span, token_id, ann, app_tok, rendering.highlight_delimiter)
        )
    return "".join(parts)


def _cdata(text: str) -> str:
    return text.replace("]]>", "]]]]><![CDATA[>")


def _build_xml_example(raw_text: str, annotated_text: str) -> str:
    return (
        "<example>\n"
        f"<raw><![CDATA[{_cdata(raw_text)}]]></raw>\n"
        f"<annotated><![CDATA[{_cdata(annotated_text)}]]></annotated>\n"
        "</example>"
    )


def _build_xml_text(
    spans: list[str],
    token_ids: list[int],
    firings: list[bool],
    per_token_activations: list[dict[str, float]],
    app_tok: AppTokenizer,
    rendering: XmlExamplesConfig,
) -> str:
    parts: list[str] = []
    for token_id, span, active, acts in zip(
        token_ids, spans, firings, per_token_activations, strict=True
    ):
        if not active:
            parts.append(span)
            continue
        ann = _fmt_ann(acts) if rendering.annotation_style == "activation" else None
        parts.append(
            _xml_token(
                span,
                token_id,
                ann,
                app_tok,
                rendering.highlight_delimiter,
                sanitize=rendering.sanitize_highlighted,
            )
        )
    return "".join(parts)


def build_annotated_examples(
    component: ComponentData,
    app_tok: AppTokenizer,
    max_examples: int,
    rendering: ExampleRenderingConfig,
    shift_firings: bool = False,
) -> Md:
    if isinstance(rendering, LegacyDelimitedExamplesConfig):
        return _build_legacy_examples(component, app_tok, max_examples, shift_firings)

    items: list[str] = []
    for ex in component.activation_examples[:max_examples]:
        if not any(ex.firings):
            continue
        firings = [False] + ex.firings[:-1] if shift_firings else ex.firings
        act_keys = list(ex.activations.keys())
        per_token_acts = [
            {k: ex.activations[k][i] for k in act_keys} for i in range(len(ex.token_ids))
        ]

        match rendering:
            case SingleLineExamplesConfig():
                items.append(
                    _build_single_line_text(
                        spans=app_tok.get_spans(ex.token_ids),
                        token_ids=ex.token_ids,
                        firings=firings,
                        per_token_activations=per_token_acts,
                        app_tok=app_tok,
                        rendering=rendering,
                    )
                )
            case XmlExamplesConfig():
                raw_spans = (
                    app_tok.get_spans(ex.token_ids)
                    if rendering.sanitize_raw
                    else app_tok.get_raw_spans(ex.token_ids)
                )
                highlighted_spans = (
                    app_tok.get_spans(ex.token_ids)
                    if rendering.sanitize_highlighted
                    else app_tok.get_raw_spans(ex.token_ids)
                )
                items.append(
                    _build_xml_example(
                        raw_text="".join(raw_spans),
                        annotated_text=_build_xml_text(
                            spans=highlighted_spans,
                            token_ids=ex.token_ids,
                            firings=firings,
                            per_token_activations=per_token_acts,
                            app_tok=app_tok,
                            rendering=rendering,
                        ),
                    )
                )
    md = Md()
    if items:
        md.numbered(items)
    return md
