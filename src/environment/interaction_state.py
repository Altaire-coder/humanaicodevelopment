from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .model import HumanState, RecursiveModelConfig

def _clip(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))

@dataclass
class LocalAIState:
    context_contamination: float = 0.0
    memory_strength: float = 0.60
    local_response_diversity: float = 1.0
    local_alignment_conditioning: float = 1.0
    local_error_correction_tendency: float = 0.50
    current_intention_anchor: float = 1.0
    session_interaction_count: int = 0

@dataclass(frozen=True)
class InteractionUpdateResult:
    learning_gain: float
    reliance_change: float
    independent_performance_change: float
    error_internalization_change: float
    context_contamination_change: float
    local_diversity_change: float

def update_interaction_state(*, human: HumanState, local_ai: LocalAIState, interaction, config: RecursiveModelConfig) -> InteractionUpdateResult:
    old_knowledge = human.knowledge
    old_reliance = human.ai_reliance
    old_independent = human.independent_performance
    old_internalization = human.error_internalization
    old_context = local_ai.context_contamination
    old_diversity = local_ai.local_response_diversity

    quality_gain = interaction.final_validity - interaction.human_input_validity
    verified = interaction.human_action in {"verify", "correct", "challenge"}
    learning_signal = max(0.0, quality_gain) * (1.0 if verified else 0.45)
    displacement_signal = (
        interaction.ai_origin_share
        * (1.0 if interaction.human_action == "accept" else 0.35)
        * max(0.0, 0.60 - interaction.final_validity)
    )

    human.knowledge = _clip(
        human.knowledge
        + config.human_learning_rate * learning_signal
        - config.human_dependence_rate * displacement_signal
    )
    human.independent_performance = _clip(
        human.independent_performance
        + 0.60 * config.human_learning_rate * learning_signal
        - config.human_dependence_rate
        * interaction.ai_origin_share
        * (1.0 if not verified else 0.20)
    )
    human.ai_reliance = _clip(
        human.ai_reliance
        + 0.04 * interaction.ai_origin_share
        * (1.0 if interaction.human_action == "accept" else -0.25)
        - 0.03 * float(verified)
    )
    human.error_internalization = _clip(
        (1.0 - config.error_persistence) * human.error_internalization
        + config.error_persistence
        * interaction.error_severity
        * interaction.ai_origin_share
        * (1.0 if interaction.human_action == "accept" else 0.25)
    )
    human.validity = interaction.final_validity
    human.novelty = interaction.final_novelty
    human.alignment = interaction.alignment
    human.confidence = interaction.final_confidence

    contamination_input = interaction.error_severity * interaction.ai_origin_share * local_ai.memory_strength
    local_ai.context_contamination = _clip(
        local_ai.context_contamination * (1.0 - config.memory_decay)
        + config.context_contamination_rate * contamination_input
    )
    local_ai.local_response_diversity = _clip(
        local_ai.local_response_diversity
        + 0.03 * interaction.valid_novelty
        - 0.04 * interaction.repetition_score
    )
    local_ai.local_alignment_conditioning = _clip(
        0.80 * local_ai.local_alignment_conditioning + 0.20 * interaction.alignment
    )
    local_ai.local_error_correction_tendency = _clip(
        0.85 * local_ai.local_error_correction_tendency
        + 0.15 * float(interaction.error_corrected)
    )
    local_ai.current_intention_anchor = _clip(
        0.90 * local_ai.current_intention_anchor + 0.10 * interaction.alignment
    )
    local_ai.session_interaction_count += 1

    return InteractionUpdateResult(
        learning_gain=human.knowledge-old_knowledge,
        reliance_change=human.ai_reliance-old_reliance,
        independent_performance_change=human.independent_performance-old_independent,
        error_internalization_change=human.error_internalization-old_internalization,
        context_contamination_change=local_ai.context_contamination-old_context,
        local_diversity_change=local_ai.local_response_diversity-old_diversity,
    )
