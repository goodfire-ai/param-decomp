"""Strategy dispatch: routes AutointerpConfig variants to their implementations."""

from spd.app.backend.app_tokenizer import AppTokenizer
from spd.autointerp.config import CompactSkepticalConfig, DualViewConfig, StrategyConfig
from spd.autointerp.schemas import ModelMetadata
from spd.autointerp.strategies.compact_skeptical import (
    format_prompt as compact_skeptical_prompt,
)
from spd.autointerp.strategies.dual_view import (
    format_prompt as dual_view_prompt,
)
from spd.harvest.analysis import TokenPRLift
from spd.harvest.schemas import ComponentData

INTERPRETATION_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string"},
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
        "reasoning": {"type": "string"},
    },
    "required": ["label", "confidence", "reasoning"],
    "additionalProperties": False,
}


def format_prompt(
    strategy: StrategyConfig,
    component: ComponentData,
    model_metadata: ModelMetadata,
    app_tok: AppTokenizer,
    input_token_stats: TokenPRLift,
    output_token_stats: TokenPRLift,
    context_tokens_per_side: int,
) -> str:
    match strategy:
        case CompactSkepticalConfig():
            return compact_skeptical_prompt(
                strategy,
                component,
                model_metadata,
                app_tok,
                input_token_stats,
                output_token_stats,
                context_tokens_per_side,
            )
        case DualViewConfig():
            return dual_view_prompt(
                strategy,
                component,
                model_metadata,
                app_tok,
                input_token_stats,
                output_token_stats,
                context_tokens_per_side,
            )
