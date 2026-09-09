from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

HumanAction = Literal[
    "accept",
    "verify",
    "correct",
    "challenge",
    "reject",
    "explore",
]

MutationType = Literal[
    "none",
    "neutral_paraphrase",
    "beneficial_elaboration",
    "beneficial_recombination",
    "corrective_mutation",
    "deleterious_distortion",
    "factual_error",
    "alignment_drift",
    "catastrophic_contamination",
]


@dataclass(frozen=True)
class RevisionResult:
    final_validity: float
    final_novelty: float
    final_alignment: float
    final_utility: float
    final_confidence: float
    creativity: float
    technical_rigor: float
    repetition_score: float
    distortion_score: float
    error_severity: float
    error_corrected: bool
    mutation_type: MutationType
    human_origin_share: float
    ai_origin_share: float


def _clip(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def revise_idea(
    *,
    action: HumanAction,
    human_input_validity: float,
    human_input_novelty: float,
    human_input_alignment: float,
    ai_output_validity: float,
    ai_output_novelty: float,
    ai_output_alignment: float,
    ai_confidence: float,
    human_knowledge: float,
    human_verification: float,
    human_creativity_preference: float,
    human_rigor_preference: float,
    rng: np.random.Generator,
) -> RevisionResult:
    # Produce the final co-created idea.
    base_weights = {
        "accept": (0.20, 0.80),
        "verify": (0.45, 0.55),
        "correct": (0.75, 0.25),
        "challenge": (0.55, 0.45),
        "reject": (0.95, 0.05),
        "explore": (0.50, 0.50),
    }
    human_w, ai_w = base_weights[action]

    correction_strength = _clip(
        0.55 * human_knowledge + 0.45 * human_verification
    )
    exploration_strength = _clip(
        0.65 * human_creativity_preference + 0.35 * (1.0 - human_rigor_preference)
    )
    rigor_strength = _clip(
        0.60 * human_rigor_preference + 0.40 * human_knowledge
    )

    weighted_validity = (
        human_w * human_input_validity + ai_w * ai_output_validity
    )
    weighted_novelty = (
        human_w * human_input_novelty + ai_w * ai_output_novelty
    )
    weighted_alignment = (
        human_w * human_input_alignment + ai_w * ai_output_alignment
    )

    if action == "correct":
        final_validity = weighted_validity + 0.30 * correction_strength
        final_alignment = weighted_alignment + 0.15 * correction_strength
        final_novelty = weighted_novelty + 0.08 * exploration_strength
    elif action == "verify":
        final_validity = weighted_validity + 0.20 * correction_strength
        final_alignment = weighted_alignment + 0.10 * correction_strength
        final_novelty = weighted_novelty
    elif action == "challenge":
        final_validity = weighted_validity + 0.08 * rigor_strength
        final_alignment = weighted_alignment + 0.05 * rigor_strength
        final_novelty = weighted_novelty + 0.22 * exploration_strength
    elif action == "explore":
        final_validity = weighted_validity - 0.05 * exploration_strength
        final_alignment = weighted_alignment - 0.03 * exploration_strength
        final_novelty = weighted_novelty + 0.30 * exploration_strength
    elif action == "reject":
        final_validity = human_input_validity
        final_alignment = human_input_alignment
        final_novelty = human_input_novelty + 0.04 * exploration_strength
    else:  # accept
        final_validity = weighted_validity
        final_alignment = weighted_alignment
        final_novelty = weighted_novelty

    noise = rng.normal(0.0, 0.015)
    final_validity = _clip(final_validity + noise)
    final_novelty = _clip(final_novelty + rng.normal(0.0, 0.015))
    final_alignment = _clip(final_alignment + rng.normal(0.0, 0.010))

    error_before = 1.0 - min(human_input_validity, ai_output_validity)
    error_after = 1.0 - final_validity
    error_corrected = error_after + 0.05 < error_before

    repetition_score = _clip(
        1.0
        - abs(final_novelty - human_input_novelty)
        - 0.50 * abs(final_novelty - ai_output_novelty)
    )
    distortion_score = _clip(
        0.55 * (1.0 - final_alignment)
        + 0.45 * max(0.0, max(human_input_validity, ai_output_validity) - final_validity)
    )
    error_severity = _clip(
        0.65 * (1.0 - final_validity) + 0.35 * distortion_score
    )

    creativity = _clip(final_novelty * final_validity * final_alignment)
    technical_rigor = _clip(
        final_validity * (0.65 + 0.35 * rigor_strength)
    )
    final_utility = _clip(
        0.40 * final_validity
        + 0.25 * final_alignment
        + 0.20 * creativity
        + 0.15 * technical_rigor
        - 0.25 * error_severity
    )
    final_confidence = _clip(
        0.55 * ai_confidence
        + 0.30 * final_validity
        + 0.15 * human_knowledge
    )

    human_w = _clip(human_w)
    ai_w = _clip(ai_w)
    total = human_w + ai_w
    human_share = human_w / total
    ai_share = ai_w / total

    if error_corrected:
        mutation_type: MutationType = "corrective_mutation"
    elif final_validity < 0.35 and final_alignment < 0.35:
        mutation_type = "catastrophic_contamination"
    elif final_validity + 0.05 < max(human_input_validity, ai_output_validity):
        mutation_type = "deleterious_distortion"
    elif final_novelty > max(human_input_novelty, ai_output_novelty) + 0.08:
        mutation_type = "beneficial_elaboration"
    elif repetition_score > 0.85:
        mutation_type = "neutral_paraphrase"
    elif final_validity < 0.50:
        mutation_type = "factual_error"
    else:
        mutation_type = "beneficial_recombination"

    return RevisionResult(
        final_validity=final_validity,
        final_novelty=final_novelty,
        final_alignment=final_alignment,
        final_utility=final_utility,
        final_confidence=final_confidence,
        creativity=creativity,
        technical_rigor=technical_rigor,
        repetition_score=repetition_score,
        distortion_score=distortion_score,
        error_severity=error_severity,
        error_corrected=error_corrected,
        mutation_type=mutation_type,
        human_origin_share=float(human_share),
        ai_origin_share=float(ai_share),
    )
