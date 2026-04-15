"""Canon interpretation strategy.

Detailed SPD-specific prompt with full decomposition explanation, sign convention,
CI-vs-act guidance, output PMI, and XML dual-view examples (raw + annotated).
"""

from spd.app.backend.app_tokenizer import AppTokenizer
from spd.autointerp.config import CANON_RENDERING, CanonConfig
from spd.autointerp.prompt_helpers import (
    DATASET_DESCRIPTIONS,
    build_annotated_examples,
    human_layer_desc,
    token_stats_section,
)
from spd.autointerp.schemas import ModelMetadata
from spd.harvest.analysis import TokenPRLift
from spd.harvest.schemas import ComponentData
from spd.utils.markdown import Md


def format_prompt(
    config: CanonConfig,
    component: ComponentData,
    model_metadata: ModelMetadata,
    app_tok: AppTokenizer,
    input_token_stats: TokenPRLift | None,
    output_token_stats: TokenPRLift | None,
    context_tokens_per_side: int,
    activation_threshold: float,
) -> str:
    rendering = CANON_RENDERING

    rate_str = (
        f"~1 in {int(1 / component.firing_density)} tokens"
        if component.firing_density > 0.0
        else "extremely rare"
    )
    canonical = model_metadata.layer_descriptions.get(component.layer, component.layer)
    layer_desc = human_layer_desc(canonical, model_metadata.n_blocks)
    dataset_desc = DATASET_DESCRIPTIONS.get(
        model_metadata.dataset_name, model_metadata.dataset_name
    )
    md = Md()

    # --- SPD method explanation ---
    md.h(3, "Context")
    md.p(
        "Below you will be presented with data about a component of a neural network as "
        'isolated by a Mechanistic Interpretability technique called "Stochastic Parameter '
        'Decomposition". You will be tasked with describing what the component does based on '
        "the evidence provided below."
    )
    md.p(
        "In Stochastic Parameter Decomposition, each weight matrix of a network is decomposed "
        'into C rank-1 parts, called "subcomponents", where C is usually greater than the rank '
        "of the weight matrix. These are parameterised as U \u2022 V (dimensions `d_out` \u00d7 "
        "`d_in`). Each subcomponent represents a one-dimensional slice of the computation the "
        "weight matrix performs."
    )
    md.p("These subcomponents are learned in an unsupervised manner under 3 main losses:")
    md.bullets(
        [
            "Faithfulness: the C rank-1 subcomponent matrices must sum to the original weight "
            "matrix - this should be a direct factorisation of the original weight matrix.",
            "Minimality / Simplicity: For a given datapoint, as few subcomponents as possible "
            "should be necessary for the network. Alternatively - as many subcomponents as "
            "possible should be ablatable",
            "Reconstruction: The network should nonetheless reproduce the behaviour of the "
            "target network.",
        ]
    )
    md.p(
        'In order to facilitate this, we train a small auxiliary "Causal Importance Network" '
        "which produces a mask of Causal Importance values in (0, 1) for each datapoint/token. "
        "The minimality/simplicity loss above incentivizes this mask to be sparse."
    )

    # --- CI and activation explanation ---
    md.p("At each token position, each component has 2 values:")
    md.numbered(
        [
            "**Causal Importance (CI):** The primary measure of component activity. CI is how "
            "causally important this component is for the model's output at this token \u2014 i.e. "
            'how much would ablating it change the output. CI is the basis for "firing": a '
            "component fires when its CI exceeds a threshold. **When interpreting a component, CI "
            "is the signal you should weight most heavily.**",
            "**Inner Activation (act):** The dot product of the input with the component's read "
            "direction (`x @ V`), scaled by the write direction norm. This measures how much the "
            "input aligns with what the component reads, regardless of whether that alignment "
            "matters for the output.",
        ]
    )
    md.p(
        "These two values are correlated but meaningfully different. A large inner activation "
        "with low CI means the input happens to align with the component's read direction, but "
        "the component's contribution isn't needed at this token. **When CI and act diverge, "
        "trust CI.** A position with high CI and low act can be genuinely important; a position "
        "with low CI and high act suggests that the component would produce an output here but "
        "that it's not necessary. When building your interpretation, weight "
        "high-CI examples heavily and treat low-CI positions as background noise, even if their "
        "act values are large."
    )

    # --- Threshold ---
    md.p(
        'A component is said to "fire" when its causal importance exceeds a threshold. '
        f"The data below uses a CI threshold of {activation_threshold}."
    )

    # --- Sign convention ---
    md.p(
        "**Sign convention:** Negating both u\u1d62 and v\u1d62 produces the same rank-1 "
        'matrix, so the absolute sign of inner activations is arbitrary \u2014 "positive" '
        'does not inherently mean excitation and "negative" does not inherently mean '
        "suppression. However, **relative sign within a component is meaningful**: tokens with "
        "positive act and tokens with negative act produce opposite contributions to the "
        "output. So if you notice examples splitting into a positive-act cluster and a "
        "negative-act cluster, that split is real and likely reflects two distinct roles (e.g. "
        "one token class activates positively, another negatively). You should describe both "
        'clusters if they exist, but do not assign meaning to which cluster is "positive" vs '
        '"negative" \u2014 only that they are opposite.'
    )

    # --- This component ---
    md.h(3, "This component")
    md.p(
        f"The component you will be labeling today comes from a decomposition of a "
        f"{model_metadata.n_blocks}-block transformer trained on {dataset_desc}. "
        f"Specifically, it is part of the {layer_desc}. "
        f"It has a firing rate of {component.firing_density * 100:.2f}% "
        f"(fires {rate_str})."
    )

    # --- Token statistics ---
    md.h(2, "Evidence:")
    md.p(
        "At each position where the component fires, we look at the model's next-token "
        "prediction distribution."
    )
    if output_token_stats is not None:
        md.extend(
            token_stats_section(
                output_token_stats,
                "Output",
                "what the model produces when this component fires",
            )
        )
    if input_token_stats is not None:
        md.extend(
            token_stats_section(
                input_token_stats,
                "Input",
                "tokens at positions where this component fires",
            )
        )

    # --- Activating examples ---
    md.h(3, "Activating examples")
    md.p(
        "The following **activating examples** are sampled uniformly at random from all "
        "positions in the dataset where the component fires (CI above threshold). For each "
        "sampled activation location, we extract a window centered on the "
        f"firing position, with up to {context_tokens_per_side} tokens of context on each "
        "side. Windows are truncated at sequence boundaries \u2014 so a firing at the beginning "
        "of a training sequence will have little or no left context. This truncation is itself "
        "evidence (e.g. a component that consistently fires near the start of sequences). We "
        "include annotations for **all** firing positions in the window - not just the firing "
        "which was sampled to produce the window, however we don't include inner activations "
        "for all tokens - this would be too noisy - all tokens have at least epsilon inner "
        "activation on almost all components."
    )
    md.p(
        "The training data consists of variable-length documents concatenated with "
        "`<|endoftext|>` separator tokens between them, then sliced into fixed "
        f"{model_metadata.seq_len}-token sequences. This means `<|endoftext|>` tokens can "
        "appear anywhere within a sequence (not just at the start), and a single sequence may "
        "contain parts of multiple documents. If you see `<|endoftext|>` in examples, it is a "
        "literal token the model processed, not a formatting artifact."
    )

    # Example format description
    md.p("Each example is shown as an XML block with two views:")
    md.bullets(
        [
            "`<raw>`: the literal token text of the window",
            "`<annotated>`: the same window with firing tokens wrapped as "
            "`[[[token (ci:X, act:Y)]]]`. When multiple consecutive tokens fire, they are grouped: "
            "`[[[token1 (ci:X1, act:Y1) token2 (ci:X2, act:Y2)]]]`",
        ]
    )

    # Annotation legend
    md.p("**Annotation legend:**")
    md.bullets(
        [
            "**ci** (causal importance): 0\u20131. How essential this component is at this "
            "position. This is the primary signal \u2014 high CI means the component genuinely "
            "matters here.",
            "**act** (inner activation): alignment of the input with the component's read "
            "direction. See the sign convention note above \u2014 relative sign differences within "
            "a component are meaningful.",
        ]
    )

    # Examples
    examples = build_annotated_examples(
        component,
        app_tok,
        config.max_examples,
        rendering=rendering,
    )
    md.extend(examples)

    # --- Task ---
    md.h(2, "Task")
    md.p(
        f"Based on all the above context and evidence, please give a label of "
        f"{config.label_max_words} words or fewer for this component. The label should "
        "describe the most salient aspect of this component's behavior. Different components "
        "are best described differently — some by what input patterns they respond to, some by "
        "what output tokens they promote, some by what contexts they appear in, some by what "
        "they *don't* fire on. Lead with whatever is most distinctive and informative for this "
        "particular component."
    )
    md.p(
        "Please also provide a short summary of your reasoning. "
        "Be epistemically honest \u2014 express "
        "uncertainty when the evidence is weak, ambiguous, or mixed. Lowercase only."
    )

    return md.build()
