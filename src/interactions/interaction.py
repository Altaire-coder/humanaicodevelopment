from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal
from uuid import uuid4

import numpy as np

from .correction import revise_idea
from .evaluation import HumanAction, evaluate_ai_response

Mechanism = Literal[
    "valid_reinforcement_or_elaboration",
    "ai_originated_contamination",
    "ai_assisted_correction",
    "mutual_error_reinforcement",
]


@dataclass(frozen=True)
class InteractionConfig:
    # Parameters for one human–AI interaction.

    accuracy_threshold: float = 0.60
    human_knowledge: float = 0.70
    human_verification: float = 0.50
    human_confirmation_bias: float = 0.30
    human_ai_reliance: float = 0.50
    human_creativity_preference: float = 0.55
    human_rigor_preference: float = 0.70
    ai_agreement: float = 0.50
    forced_action: HumanAction | None = None


@dataclass(frozen=True)
class InteractionResult:
    # Structured output emitted by a single interaction cycle.

    run_id: str
    scenario_id: str
    interaction_id: str
    event_id: str
    time: int
    generation: int
    human_id: str
    ai_id: str
    parent_idea_id: str
    new_idea_id: str
    lineage_id: str

    human_input_validity: float
    human_input_novelty: float
    human_input_alignment: float
    ai_output_validity: float
    ai_output_novelty: float
    ai_output_alignment: float
    ai_confidence: float

    human_action: HumanAction
    detected_error: bool
    mechanism: Mechanism
    error_introduced_by: str | None
    error_corrected: bool

    final_validity: float
    final_novelty: float
    alignment: float
    utility: float
    final_confidence: float
    creativity: float
    technical_rigor: float
    repetition_score: float
    distortion_score: float
    error_severity: float
    mutation_type: str
    human_origin_share: float
    ai_origin_share: float

    complementarity: float
    valid_novelty: float
    co_thinking_score: float
    idea_similarity: float
    co_movement_score: float
    constructive_co_movement: bool
    alive: bool

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


def _clip(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def classify_mechanism(
    human_validity: float,
    ai_validity: float,
    threshold: float,
) -> tuple[Mechanism, str | None]:
    human_accurate = human_validity >= threshold
    ai_accurate = ai_validity >= threshold

    if human_accurate and ai_accurate:
        return "valid_reinforcement_or_elaboration", None
    if human_accurate and not ai_accurate:
        return "ai_originated_contamination", "ai"
    if not human_accurate and ai_accurate:
        return "ai_assisted_correction", "human"
    return "mutual_error_reinforcement", "human_and_ai"


def run_interaction(
    *,
    human_input_validity: float,
    human_input_novelty: float,
    ai_output_validity: float,
    ai_output_novelty: float,
    ai_confidence: float,
    human_input_alignment: float = 1.0,
    ai_output_alignment: float = 1.0,
    config: InteractionConfig | None = None,
    rng: np.random.Generator | None = None,
    run_id: str = "R_0001",
    scenario_id: str = "S_stage1",
    time: int = 0,
    generation: int = 1,
    human_id: str = "H_0001",
    ai_id: str = "A_0001",
    parent_idea_id: str = "I_000001",
    lineage_id: str = "L_0001",
    new_idea_id: str | None = None,
) -> InteractionResult:
    
    # Execute one complete cycle: HumanInput_t -> AIResponse_t -> HumanEvaluation_t -> RevisedIdea_t
    # Co-thinking is measured as productive integration.
    # Co-movement is measured as similarity/convergence adjusted by quality gain.
    
    cfg = config or InteractionConfig()
    rng = rng or np.random.default_rng()

    values = {
        "human_input_validity": human_input_validity,
        "human_input_novelty": human_input_novelty,
        "human_input_alignment": human_input_alignment,
        "ai_output_validity": ai_output_validity,
        "ai_output_novelty": ai_output_novelty,
        "ai_output_alignment": ai_output_alignment,
        "ai_confidence": ai_confidence,
    }
    for name, value in values.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1; got {value}.")

    mechanism, error_introduced_by = classify_mechanism(
        human_input_validity,
        ai_output_validity,
        cfg.accuracy_threshold,
    )

    evaluation = evaluate_ai_response(
        human_input_validity=human_input_validity,
        ai_output_validity=ai_output_validity,
        ai_confidence=ai_confidence,
        ai_agreement=cfg.ai_agreement,
        human_verification=cfg.human_verification,
        human_confirmation_bias=cfg.human_confirmation_bias,
        human_ai_reliance=cfg.human_ai_reliance,
        human_knowledge=cfg.human_knowledge,
        human_creativity_preference=cfg.human_creativity_preference,
        rng=rng,
        forced_action=cfg.forced_action,
    )

    revision = revise_idea(
        action=evaluation.action,
        human_input_validity=human_input_validity,
        human_input_novelty=human_input_novelty,
        human_input_alignment=human_input_alignment,
        ai_output_validity=ai_output_validity,
        ai_output_novelty=ai_output_novelty,
        ai_output_alignment=ai_output_alignment,
        ai_confidence=ai_confidence,
        human_knowledge=cfg.human_knowledge,
        human_verification=cfg.human_verification,
        human_creativity_preference=cfg.human_creativity_preference,
        human_rigor_preference=cfg.human_rigor_preference,
        rng=rng,
    )

    # Complementarity: final utility over the stronger initial actor quality.
    strongest_initial_quality = max(
        human_input_validity * human_input_alignment,
        ai_output_validity * ai_output_alignment,
    )
    complementarity = _clip(revision.final_utility - strongest_initial_quality + 0.50)

    valid_novelty = _clip(
        revision.final_novelty
        * revision.final_validity
        * revision.final_alignment
    )

    perspective_expansion = _clip(
        max(
            0.0,
            revision.final_novelty
            - max(human_input_novelty, ai_output_novelty),
        )
        * 4.0
    )

    revision_depth = _clip(
        abs(revision.final_validity - human_input_validity)
        + abs(revision.final_novelty - human_input_novelty)
        + abs(revision.final_alignment - human_input_alignment)
    )

    co_thinking_score = _clip(
        0.30 * complementarity
        + 0.30 * valid_novelty
        + 0.20 * revision_depth
        + 0.20 * perspective_expansion
        - 0.20 * revision.repetition_score
    )

    # with embedding-based semantic similarity.
    idea_distance = np.sqrt(
        (human_input_validity - ai_output_validity) ** 2
        + (human_input_novelty - ai_output_novelty) ** 2
        + (human_input_alignment - ai_output_alignment) ** 2
    ) / np.sqrt(3.0)
    idea_similarity = _clip(1.0 - idea_distance)

    initial_joint_quality = (
        0.50 * human_input_validity * human_input_alignment
        + 0.50 * ai_output_validity * ai_output_alignment
    )
    final_joint_quality = revision.final_validity * revision.final_alignment
    quality_gain = final_joint_quality - initial_joint_quality
    co_movement_score = float(idea_similarity * quality_gain)
    constructive_co_movement = bool(co_movement_score > 0.0)

    alive = bool(
        revision.final_validity >= 0.20
        and revision.final_alignment >= 0.20
        and revision.final_utility >= 0.15
    )

    return InteractionResult(
        run_id=run_id,
        scenario_id=scenario_id,
        interaction_id=f"INT_{uuid4().hex[:10]}",
        event_id=f"E_{uuid4().hex[:10]}",
        time=time,
        generation=generation,
        human_id=human_id,
        ai_id=ai_id,
        parent_idea_id=parent_idea_id,
        new_idea_id=new_idea_id or f"I_{uuid4().hex[:10]}",
        lineage_id=lineage_id,
        human_input_validity=human_input_validity,
        human_input_novelty=human_input_novelty,
        human_input_alignment=human_input_alignment,
        ai_output_validity=ai_output_validity,
        ai_output_novelty=ai_output_novelty,
        ai_output_alignment=ai_output_alignment,
        ai_confidence=ai_confidence,
        human_action=evaluation.action,
        detected_error=evaluation.detected_error,
        mechanism=mechanism,
        error_introduced_by=error_introduced_by,
        error_corrected=revision.error_corrected,
        final_validity=revision.final_validity,
        final_novelty=revision.final_novelty,
        alignment=revision.final_alignment,
        utility=revision.final_utility,
        final_confidence=revision.final_confidence,
        creativity=revision.creativity,
        technical_rigor=revision.technical_rigor,
        repetition_score=revision.repetition_score,
        distortion_score=revision.distortion_score,
        error_severity=revision.error_severity,
        mutation_type=revision.mutation_type,
        human_origin_share=revision.human_origin_share,
        ai_origin_share=revision.ai_origin_share,
        complementarity=complementarity,
        valid_novelty=valid_novelty,
        co_thinking_score=co_thinking_score,
        idea_similarity=idea_similarity,
        co_movement_score=co_movement_score,
        constructive_co_movement=constructive_co_movement,
        alive=alive,
    )
