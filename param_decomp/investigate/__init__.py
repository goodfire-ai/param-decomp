"""Investigation: SLURM-based agent investigation of model behaviors.

This module provides infrastructure for launching a Claude Code agent to investigate
behaviors in a PD model decomposition. Each investigation:
1. Starts an isolated app backend instance (separate database, unique port)
2. Receives a specific research question and detailed instructions
3. Investigates behaviors and writes findings to append-only JSONL files
"""

from param_decomp.investigate.schemas import (
    BehaviorExplanation,
    ComponentInfo,
    Evidence,
    InvestigationEvent,
)

__all__ = [
    "BehaviorExplanation",
    "ComponentInfo",
    "Evidence",
    "InvestigationEvent",
]
