"""Rich examples interpretation strategy.

Drops token statistics entirely. Shows per-token CI and activation values inline
in the examples so the LLM can judge evidence quality directly.
"""

from spd.app.backend.app_tokenizer import AppTokenizer
from spd.autointerp.config import RichExamplesConfig
from spd.autointerp.prompt_helpers import (
    DATASET_DESCRIPTIONS,
    build_annotated_examples,
    describe_example_rendering,
    human_layer_desc,
)
from spd.autointerp.schemas import DecompositionMethod, ModelMetadata
from spd.harvest.analysis import TokenPRLift
from spd.harvest.schemas import ComponentData
from spd.utils.markdown import Md

_DECOMPOSITION_DESCRIPTIONS: dict[DecompositionMethod, str] = {
    "spd": (
        "Each component is a rank-1 parameter matrix learned by Stochastic Parameter "
        "Decomposition (SPD). A weight matrix W is decomposed as W ≈ Σ u_i v_i^T. "
        "When the model processes a token, each component computes an activation: the inner "
        "product of the residual stream with its read direction v_i. This value can be "
        "positive or negative depending on how the input aligns with v_i — the sign is an "
        "arbitrary consequence of how the vectors were initialised and does not by itself "
        "mean suppression. CI and activation magnitude are the main indicators of whether "
        "the component is active at a position, but within one component the sign can still "
        "separate distinct patterns. "
        "Each component also has a causal importance (CI) value per token position: CI near 1 "
        "means the component is essential at that position, CI near 0 means it can be ablated "
        "without affecting output. A component 'fires' when its CI is high."
    ),
    "clt": (
        "Each component is a feature from a Cross-Layer Transcoder (CLT). CLTs learn sparse, "
        "interpretable features that map activations at one layer to contributions at another. "
        "A component 'fires' when its activation magnitude is high."
    ),
    "transcoder": (
        "Each component is a feature from a Transcoder, which learns a sparse dictionary of "
        "linear transformations mapping MLP inputs to MLP outputs. A component 'fires' when "
        "its encoder activation is above threshold."
    ),
}


def format_prompt(
    config: RichExamplesConfig,
    component: ComponentData,
    model_metadata: ModelMetadata,
    app_tok: AppTokenizer,
    output_token_stats: TokenPRLift,
    context_tokens_per_side: int,
) -> str:
    fires_on = build_annotated_examples(
        component,
        app_tok,
        config.max_examples,
        rendering=config.example_rendering,
    )

    rate_str = (
        f"~1 in {int(1 / component.firing_density)} tokens"
        if component.firing_density > 0.0
        else "extremely rare"
    )

    canonical = model_metadata.layer_descriptions.get(component.layer, component.layer)
    layer_desc = human_layer_desc(canonical, model_metadata.n_blocks)
    model_name = model_metadata.model_class.rsplit(".", 1)[-1]

    dataset_line = ""
    if config.include_dataset_description:
        dataset_desc = DATASET_DESCRIPTIONS.get(
            model_metadata.dataset_name, model_metadata.dataset_name
        )
        dataset_line = f", dataset: {dataset_desc}"

    md = Md()
    md.p("Describe what this neural network component does.")
    md.p(
        "Each component has an input function (what causes it to fire) and an output "
        "function (what tokens it causes the model to produce). These are often different "
        "— a component might fire on periods but produce sentence-opening words, or "
        "fire on prepositions but produce abstract nouns."
    )
    md.p(
        "The activation examples are sampled and may not be fully representative. "
        "Look for patterns that are consistent across multiple examples, and express "
        "uncertainty when the evidence is weak, noisy, or clearly mixed."
    )

    md.h(2, "Context")
    md.bullets(
        [
            f"Model: {model_name} ({model_metadata.n_blocks} blocks){dataset_line}",
            f"Component location: {layer_desc}",
            f"Component firing rate: {component.firing_density * 100:.2f}% ({rate_str})",
        ]
    )

    md.h(2, "Data presentation")
    md.extend(
        _build_data_section(
            model_metadata.seq_len, context_tokens_per_side, model_metadata.decomposition_method
        )
    )

    md.h(3, "Example annotation format")
    md.p(describe_example_rendering(config.example_rendering))
    _build_annotation_legend(md, component)

    if output_token_stats.top_pmi:
        md.h(2, "Output evidence")
        md.p(
            "Top output PMI tokens, filtered to ignore extremely low-support predictions. "
            "These are tokens the model disproportionately predicts when this component fires."
        )
        md.labeled_list(
            "**Output PMI:**",
            [f"{repr(tok)}: {pmi:.2f}" for tok, pmi in output_token_stats.top_pmi[:10]],
        )

    md.h(2, "Activation examples — where the component fires")
    if config.example_rendering.format == "xml":
        md.p(
            "Each example shows both the raw window and a highlighted version with inline "
            "activation values on firing tokens."
        )
    else:
        md.p(
            "Each firing token shows its activation values inline. "
            "Use these to judge how strongly the component responds at each position."
        )
    md.extend(fires_on)

    md.h(2, "Task")
    md.p(
        f"Give a {config.label_max_words}-word-or-fewer label describing this component's "
        "function. The label should read like a short description of the job this component "
        "does in the network. Use both the activation examples and the output evidence."
    )
    md.p(
        "Be epistemically honest — express uncertainty in the label "
        "when the evidence is weak, ambiguous, or mixed. Lowercase only."
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
    md.p(_DECOMPOSITION_DESCRIPTIONS[decomposition_method])
    md.h(3, "Data")
    md.p(
        f"The model processes sequences of {seq_len} tokens. "
        f"Each activation example below shows a {window_size}-token window centered on the "
        f"firing token, with up to {context_tokens_per_side} tokens of context on each side. "
        f"Windows are truncated at sequence boundaries. "
        f"Examples are sampled uniformly at random from all firings across the dataset. "
        "If a firing token appears at the left or right edge of the shown window, that may "
        "itself be evidence of a boundary or beginning-of-sequence feature."
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
            "**act** (component activation): Inner product with the component's read direction. "
            "The global sign convention is arbitrary — the component would be equivalent with "
            "both vectors negated — but sign is meaningful within a component's examples. "
            "Positive and negative activations produce opposite contributions to the output, "
            "and may represent qualitatively distinct input patterns rather than just more/less "
            "of the same thing (e.g. negative acts on one token class, positive on another). "
            "Check whether examples cluster by sign, and if so, treat each polarity as "
            "potentially a separate input pattern worth describing independently."
        )
    if "activation" in act_keys:
        legend_items.append(
            "**act** (activation): The component's activation magnitude. "
            "Larger values mean stronger signal."
        )
    if legend_items:
        md.bullets(legend_items)
    md.p(
        "Example: `the [[[cat]]] (ci:0.92, act:0.45) sat` or `the <<<cat (ci:0.92, act:0.45)>>> sat` "
        "— 'cat' is a firing token."
    )
