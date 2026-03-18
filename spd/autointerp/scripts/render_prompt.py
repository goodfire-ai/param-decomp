"""Render a prompt from dummy data for prompt engineering iteration.

Usage:
    python -m spd.autointerp.scripts.render_prompt
"""

from spd.app.backend.app_tokenizer import AppTokenizer
from spd.autointerp.config import RichExamplesConfig
from spd.autointerp.schemas import ModelMetadata
from spd.autointerp.strategies.rich_examples import format_prompt
from spd.harvest.schemas import ActivationExample, ComponentData, ComponentTokenPMI

TOKENIZER_NAME = "openai-community/gpt2"

# Dummy data with negative component_activation values to exercise the suppression
# misinterpretation failure mode.
DUMMY_COMPONENT = ComponentData(
    component_key="h.2.attn.v_proj:52",
    layer="h.2.attn.v_proj",
    component_idx=52,
    mean_activations={"causal_importance": 0.04, "component_activation": -0.31},
    firing_density=0.04,
    activation_examples=[
        ActivationExample(
            token_ids=[1026, 547, 257, 3217, 290, 6792, 1110, 13, 383, 1200, 373, 1016, 284, 1057],
            firings=[
                False,
                False,
                False,
                False,
                False,
                True,
                True,
                False,
                False,
                True,
                True,
                False,
                False,
                False,
            ],
            activations={
                "causal_importance": [
                    0.02,
                    0.03,
                    0.01,
                    0.05,
                    0.02,
                    0.45,
                    0.82,
                    0.03,
                    0.04,
                    0.38,
                    0.71,
                    0.03,
                    0.02,
                    0.05,
                ],
                "component_activation": [
                    0.12,
                    0.08,
                    -0.05,
                    0.11,
                    -0.09,
                    -0.63,
                    -1.21,
                    -0.03,
                    0.07,
                    -0.54,
                    -0.98,
                    -0.14,
                    0.06,
                    0.03,
                ],
            },
        ),
        ActivationExample(
            token_ids=[383, 1200, 2540, 284, 262, 4571, 13, 366, 1639, 460, 466, 340, 553],
            firings=[
                False,
                True,
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
                True,
            ],
            activations={
                "causal_importance": [
                    0.03,
                    0.74,
                    0.02,
                    0.01,
                    0.03,
                    0.68,
                    0.02,
                    0.04,
                    0.03,
                    0.02,
                    0.04,
                    0.78,
                    0.02,
                ],
                "component_activation": [
                    0.09,
                    -1.04,
                    0.03,
                    -0.07,
                    0.11,
                    -0.91,
                    0.04,
                    -0.02,
                    0.08,
                    -0.06,
                    0.03,
                    -1.15,
                    0.01,
                ],
            },
        ),
        ActivationExample(
            token_ids=[679, 2486, 284, 534, 2563, 13, 366, 1212, 318, 407, 625],
            firings=[False, False, False, False, True, False, False, False, False, False, True],
            activations={
                "causal_importance": [
                    0.02,
                    0.03,
                    0.02,
                    0.04,
                    0.79,
                    0.03,
                    0.05,
                    0.02,
                    0.04,
                    0.03,
                    0.67,
                ],
                "component_activation": [
                    0.04,
                    0.06,
                    -0.02,
                    0.09,
                    -0.87,
                    -0.05,
                    0.03,
                    0.07,
                    -0.11,
                    0.04,
                    -0.72,
                ],
            },
        ),
    ],
    input_token_pmi=ComponentTokenPMI(
        top=[(13, 2.1), (366, 1.8), (553, 1.6), (625, 1.5)],
        bottom=[(257, -1.2), (290, -0.9)],
    ),
    output_token_pmi=ComponentTokenPMI(
        top=[(679, 2.4), (1212, 2.0), (1639, 1.9)],
        bottom=[(257, -1.1)],
    ),
)

DUMMY_MODEL_METADATA = ModelMetadata(
    n_blocks=4,
    model_class="pile_llama_simple_mlp",
    dataset_name="danbraunai/pile-uncopyrighted-tok-shuffled",
    layer_descriptions={"h.2.attn.v_proj": "2.attn.v"},
    seq_len=512,
    decomposition_method="spd",
)

DUMMY_CONFIG = RichExamplesConfig(
    max_examples=30,
    include_dataset_description=True,
    label_max_words=8,
    forbidden_words=None,
)


def main() -> None:
    app_tok = AppTokenizer.from_pretrained(TOKENIZER_NAME)
    prompt = format_prompt(
        config=DUMMY_CONFIG,
        component=DUMMY_COMPONENT,
        model_metadata=DUMMY_MODEL_METADATA,
        app_tok=app_tok,
        context_tokens_per_side=10,
    )
    print(prompt)


if __name__ == "__main__":
    main()
