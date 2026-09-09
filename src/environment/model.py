from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RecursiveModelConfig:
    num_steps: int = 50
    seed: int = 42

    # Initial idea state
    initial_validity: float = 0.75
    initial_novelty: float = 0.55
    initial_alignment: float = 1.00
    initial_human_confidence: float = 0.70

    # Human state
    human_knowledge: float = 0.70
    human_verification: float = 0.50
    human_confirmation_bias: float = 0.30
    human_ai_reliance: float = 0.50
    human_creativity_preference: float = 0.55
    human_rigor_preference: float = 0.70
    human_learning_rate: float = 0.08
    human_dependence_rate: float = 0.05

    # AI state
    ai_accuracy: float = 0.80
    ai_novelty_capacity: float = 0.65
    ai_alignment: float = 0.90
    ai_confidence_calibration: float = 0.75
    ai_sycophancy: float = 0.30
    ai_memory_strength: float = 0.60
    ai_personalization: float = 0.50
    ai_creativity_setting: float = 0.60
    ai_rigor_setting: float = 0.75

    # Recursive feedback
    ai_feedback_reuse: float = 0.50
    human_original_input_share: float = 0.50
    memory_decay: float = 0.05
    context_contamination_rate: float = 0.03
    error_persistence: float = 0.50

    # Platform update
    platform_learning_enabled: bool = False
    platform_update_interval: int = 10
    platform_interaction_sampling_rate: float = 0.20
    platform_learning_rate: float = 0.05

    # Thresholds
    accuracy_threshold: float = 0.60
    alive_validity_threshold: float = 0.20
    alive_alignment_threshold: float = 0.20

    # Identifiers
    run_id: str = "R_0001"
    scenario_id: str = "S0_no_treatment"
    human_id: str = "H_0001"
    ai_id: str = "A_0001"
    platform_id: str = "P_0001"
    model_id: str = "MODEL_V1"
    lineage_id: str = "L_0001"


@dataclass
class HumanState:
    validity: float
    novelty: float
    alignment: float
    confidence: float
    knowledge: float
    verification: float
    confirmation_bias: float
    ai_reliance: float
    creativity_preference: float
    rigor_preference: float
    independent_performance: float
    error_internalization: float = 0.0


@dataclass
class AIState:
    #Global/model-level AI state.
    #Conversation-level state such as context contamination and local response conditioning belongs in LocalAIState, not here. The context_contamination field remains temporarily for backward compatibility with older code and should stay unchanged during ordinary local interactions.
    accuracy: float
    novelty_capacity: float
    alignment: float
    confidence_calibration: float
    sycophancy: float
    memory_strength: float
    personalization: float
    creativity_setting: float
    rigor_setting: float

    context_contamination: float = 0.0
    model_contamination: float = 0.0
    response_diversity: float = 1.0
    model_version: str = "MODEL_V1"
    copying_fidelity: float = 0.50


@dataclass
class PlatformState:
    generation: int = 0
    model_version: str = "MODEL_V1"
    training_human_share: float = 0.60
    training_ai_share: float = 0.30
    training_verified_share: float = 0.10
    context_contamination: float = 0.0
    model_contamination: float = 0.0
    response_diversity: float = 1.0
    performance: float = 0.80
    update_count: int = 0


@dataclass
class RecursiveSimulationResult:
    """All output tables produced by one recursive run."""

    events: list[dict[str, Any]] = field(default_factory=list)
    idea_states: list[dict[str, Any]] = field(default_factory=list)
    human_states: list[dict[str, Any]] = field(default_factory=list)
    ai_states: list[dict[str, Any]] = field(default_factory=list)

    # Stage 7: local interaction state and explicit platform-learning events.
    local_ai_states: list[dict[str, Any]] = field(default_factory=list)
    platform_update_events: list[dict[str, Any]] = field(default_factory=list)

    platform_states: list[dict[str, Any]] = field(default_factory=list)
    lineage_edges: list[dict[str, Any]] = field(default_factory=list)

    def as_frames(self) -> dict[str, Any]:
        import pandas as pd

        return {
            "events": pd.DataFrame(self.events),
            "idea_states": pd.DataFrame(self.idea_states),
            "human_states": pd.DataFrame(self.human_states),
            "ai_states": pd.DataFrame(self.ai_states),
            "local_ai_states": pd.DataFrame(self.local_ai_states),
            "platform_states": pd.DataFrame(self.platform_states),
            "platform_update_events": pd.DataFrame(
                self.platform_update_events
            ),
            "lineage_edges": pd.DataFrame(self.lineage_edges),
        }
