"""Schemas for investigation outputs.

All agent outputs are append-only JSONL files. Each line is a JSON object
conforming to one of the schemas defined here.
"""

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class ComponentInfo(BaseModel):
    """Information about a component involved in a behavior."""

    component_key: str = Field(
        ...,
        description="Component key in format 'layer:component_idx' (e.g., 'h.0.mlp.c_fc:5')",
    )
    role: str = Field(
        ...,
        description="The role this component plays in the behavior (e.g., 'stores subject gender')",
    )
    interpretation: str | None = Field(
        default=None,
        description="Auto-interp label for this component if available",
    )


class Evidence(BaseModel):
    """A piece of supporting evidence for an explanation."""

    evidence_type: Literal["ablation", "attribution", "activation_pattern", "correlation", "other"]
    description: str = Field(
        ...,
        description="Description of the evidence",
    )
    details: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional structured details (e.g., ablation results, attribution values)",
    )


class BehaviorExplanation(BaseModel):
    """A candidate explanation for a behavior discovered by an agent.

    This is the primary output schema for agent investigations. Each explanation
    describes a behavior (demonstrated by a subject prompt), the components involved,
    and supporting evidence.
    """

    subject_prompt: str = Field(
        ...,
        description="A prompt that demonstrates the behavior being explained",
    )
    behavior_description: str = Field(
        ...,
        description="Clear description of the behavior (e.g., 'correctly predicts gendered pronoun')",
    )
    components_involved: list[ComponentInfo] = Field(
        ...,
        description="List of components involved in this behavior and their roles",
    )
    explanation: str = Field(
        ...,
        description="Explanation of how the components work together to produce the behavior",
    )
    supporting_evidence: list[Evidence] = Field(
        default_factory=list,
        description="Evidence supporting this explanation (ablations, attributions, etc.)",
    )
    confidence: Literal["high", "medium", "low"] = Field(
        ...,
        description="Agent's confidence in this explanation",
    )
    alternative_hypotheses: list[str] = Field(
        default_factory=list,
        description="Alternative hypotheses that were considered but not fully supported",
    )
    limitations: list[str] = Field(
        default_factory=list,
        description="Known limitations of this explanation",
    )


class InvestigationEvent(BaseModel):
    """A generic event logged by an agent during investigation.

    Used for logging progress, observations, and other non-explanation events.
    """

    event_type: Literal[
        "start",
        "progress",
        "observation",
        "hypothesis",
        "test_result",
        "explanation",
        "error",
        "complete",
    ]
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    message: str
    details: dict[str, Any] = Field(default_factory=dict)
