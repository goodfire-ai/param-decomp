"""Compact skeptical interpretation strategy.

Short labels (2-5 words), skeptical tone, structured JSON output.
Extracted from the original prompt_template.py.
"""

from spd.app.backend.app_tokenizer import AppTokenizer
from spd.autointerp.config import CompactSkepticalConfig
from spd.autointerp.prompt_helpers import (
    DATASET_DESCRIPTIONS,
    build_data_presentation,
    build_fires_on_examples,
)
from spd.autointerp.schemas import ModelMetadata
from spd.harvest.analysis import TokenPRLift
from spd.harvest.schemas import ComponentData
from spd.utils.markdown import Md


def format_prompt(
    config: CompactSkepticalConfig,
    component: ComponentData,
    model_metadata: ModelMetadata,
    app_tok: AppTokenizer,
    input_token_stats: TokenPRLift,
    output_token_stats: TokenPRLift,
    context_tokens_per_side: int,
) -> str:
    input_pmi: list[tuple[str, float]] | None = None
    output_pmi: list[tuple[str, float]] | None = None

    if config.include_pmi:
        input_pmi = [
            (app_tok.get_tok_display(tid), pmi) for tid, pmi in component.input_token_pmi.top
        ]
        output_pmi = [
            (app_tok.get_tok_display(tid), pmi) for tid, pmi in component.output_token_pmi.top
        ]

    input_section = _build_input_section(input_token_stats, input_pmi)
    output_section = _build_output_section(output_token_stats, output_pmi)
    examples_section = build_fires_on_examples(component, app_tok, config.max_examples)

    rate_str = (
        f"~1 in {int(1 / component.firing_density)} tokens"
        if component.firing_density > 0.0
        else "extremely rare"
    )

    layer_desc = model_metadata.layer_descriptions.get(component.layer, component.layer)

    dataset_line = ""
    if config.include_dataset_description:
        dataset_desc = DATASET_DESCRIPTIONS[model_metadata.dataset_name]
        dataset_line = f", dataset: {dataset_desc}"

    forbidden = ", ".join(config.forbidden_words) if config.forbidden_words else "(none)"

    md = Md()
    md.p("Label this neural network component.")

    md.h(2, "Context").bullets(
        [
            f"Model: {model_metadata.model_class} ({model_metadata.n_blocks} blocks){dataset_line}",
            f"Component location: {layer_desc}",
            f"Component firing rate: {component.firing_density * 100:.2f}% ({rate_str})",
        ]
    )

    md.h(2, "Data presentation")
    md.extend(
        build_data_presentation(
            model_metadata.seq_len, context_tokens_per_side, model_metadata.decomposition_method
        )
    )

    md.h(2, "Token correlations")
    md.extend(input_section).extend(output_section)

    md.h(2, "Activation examples (active tokens in <<delimiters>>)")
    md.extend(examples_section)

    md.h(2, "Task")
    md.p(f"Give a 2-{config.label_max_words} word label for what this component detects.")
    md.p(
        "Be SKEPTICAL. If you can't identify specific tokens or a tight grammatical "
        'pattern, say "unclear".'
    )
    md.p("Rules:")
    md.numbered(
        [
            'Good labels name SPECIFIC tokens: "\'the\'", "##ing suffix", "she/her pronouns"',
            'Say "unclear" if: tokens are too varied, pattern is abstract, or evidence is weak',
            f"FORBIDDEN words (too vague): {forbidden}",
            "Lowercase only",
            'Confidence: "high" = clear, specific pattern with strong evidence; '
            '"medium" = plausible but noisy; "low" = speculative',
        ]
    )
    md.p(
        'GOOD: "##ed suffix", "\'and\' conjunction", "she/her/hers", "period then capital", "unclear"\n'
        'BAD: "various words and punctuation", "verbs and adjectives", "tokens near commas"'
    )

    return md.build()


def _build_input_section(
    input_stats: TokenPRLift,
    input_pmi: list[tuple[str, float]] | None,
) -> Md:
    md = Md()
    if input_stats.top_precision:
        md.labeled_list(
            "**Input precision:**",
            [f"{repr(tok)}: {prec * 100:.0f}%" for tok, prec in input_stats.top_precision[:8]],
        )
    if input_pmi:
        md.labeled_list(
            "**Input PMI:**",
            [f"{repr(tok)}: {pmi:.2f}" for tok, pmi in input_pmi[:6]],
        )
    return md


def _build_output_section(
    output_stats: TokenPRLift,
    output_pmi: list[tuple[str, float]] | None,
) -> Md:
    md = Md()
    if output_stats.top_precision:
        md.labeled_list(
            "**Output precision:**",
            [f"{repr(tok)}: {prec * 100:.0f}%" for tok, prec in output_stats.top_precision[:10]],
        )
    if output_pmi:
        md.labeled_list(
            "**Output PMI:**",
            [f"{repr(tok)}: {pmi:.2f}" for tok, pmi in output_pmi[:6]],
        )
    return md
