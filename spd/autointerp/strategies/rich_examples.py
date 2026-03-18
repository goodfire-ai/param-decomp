"""Rich examples interpretation strategy.

Drops token statistics entirely. Shows per-token CI and activation values inline
in the examples so the LLM can judge evidence quality directly.
"""

from spd.app.backend.app_tokenizer import AppTokenizer
from spd.autointerp.config import RichExamplesConfig
from spd.autointerp.prompt_helpers import (
    DATASET_DESCRIPTIONS,
    build_annotated_examples,
    density_note,
    human_layer_desc,
    layer_position_note,
)
from spd.autointerp.schemas import DECOMPOSITION_DESCRIPTIONS, DecompositionMethod, ModelMetadata
from spd.harvest.schemas import ComponentData
from spd.utils.markdown import Md


def format_prompt(
    config: RichExamplesConfig,
    component: ComponentData,
    model_metadata: ModelMetadata,
    app_tok: AppTokenizer,
    context_tokens_per_side: int,
) -> str:
    fires_on = build_annotated_examples(component, app_tok, config.max_examples)

    rate_str = (
        f"~1 in {int(1 / component.firing_density)} tokens"
        if component.firing_density > 0.0
        else "extremely rare"
    )

    canonical = model_metadata.layer_descriptions.get(component.layer, component.layer)
    layer_desc = human_layer_desc(canonical, model_metadata.n_blocks)
    position_note = layer_position_note(canonical, model_metadata.n_blocks)
    dens_note = density_note(component.firing_density)

    context_notes = " ".join(filter(None, [position_note, dens_note]))

    dataset_line = ""
    if config.include_dataset_description:
        dataset_desc = DATASET_DESCRIPTIONS.get(
            model_metadata.dataset_name, model_metadata.dataset_name
        )
        dataset_line = f", dataset: {dataset_desc}"

    forbidden_sentence = (
        "FORBIDDEN vague words: " + ", ".join(config.forbidden_words) + ". "
        if config.forbidden_words
        else ""
    )

    md = Md()
    md.p("Describe what this neural network component does.")
    md.p(
        "Each component has an input function (what causes it to fire) and an output "
        "function (what tokens it causes the model to produce). These are often different "
        "— a component might fire on periods but produce sentence-opening words, or "
        "fire on prepositions but produce abstract nouns."
    )

    md.h(2, "Context")
    md.bullets(
        [
            f"Model: {model_metadata.model_class} ({model_metadata.n_blocks} blocks){dataset_line}",
            f"Component location: {layer_desc}",
            f"Component firing rate: {component.firing_density * 100:.2f}% ({rate_str})",
        ]
    )
    if context_notes:
        md.p(context_notes)

    md.h(2, "Data presentation")
    md.extend(
        _build_data_section(
            model_metadata.seq_len, context_tokens_per_side, model_metadata.decomposition_method
        )
    )

    md.h(3, "Example annotation format")
    md.p(
        "Firing tokens are wrapped in <<<triple angle brackets>>> with their activation "
        "values. Non-firing tokens appear as plain text."
    )
    _build_annotation_legend(md, component)

    md.h(2, "Activation examples — where the component fires")
    md.p(
        "Each firing token shows its activation values inline. "
        "Use these to judge how strongly the component responds at each position."
    )
    md.extend(fires_on)

    md.h(2, "Task")
    md.p(
        f"Give a {config.label_max_words}-word-or-fewer label describing this component's "
        "function. The label should read like a short description of the job this component "
        "does in the network. Use both the input and output evidence."
    )
    md.p(
        f"Be epistemically honest — express uncertainty in the label and confidence "
        f"field when the evidence is weak or ambiguous. {forbidden_sentence}Lowercase only."
    )

    return md.build()


def _build_data_section(
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
    return md


def _build_annotation_legend(md: Md, component: ComponentData) -> None:
    first_ex = next((ex for ex in component.activation_examples if any(ex.firings)), None)
    if first_ex is None:
        return
    act_keys = list(first_ex.activations.keys())
    legend_items: list[str] = []
    if "causal_importance" in act_keys:
        legend_items.append(
            "**ci** (causal importance): 0–1. How essential this component is at this position. "
            "ci near 1 = component is critical here; ci near 0 = component could be ablated."
        )
    if "component_activation" in act_keys:
        legend_items.append(
            "**act** (component activation): The component's pre-mask activation magnitude. "
            "Can be positive or negative. Larger absolute values mean stronger signal."
        )
    if "activation" in act_keys:
        legend_items.append(
            "**act** (activation): The component's activation magnitude. "
            "Larger values mean stronger signal."
        )
    if legend_items:
        md.bullets(legend_items)
    md.p("Example: `the <<<cat (ci:0.92, act:0.45)>>> sat` — 'cat' is a firing token.")
