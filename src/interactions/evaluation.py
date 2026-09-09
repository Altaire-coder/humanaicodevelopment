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


@dataclass(frozen=True)
class EvaluationResult:
    # Human evaluation of an AI response.

    action: HumanAction
    detected_error: bool
    acceptance_probability: float
    verification_probability: float
    challenge_probability: float
    perceived_ai_quality: float


def _clip(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def evaluate_ai_response(
    *,
    human_input_validity: float,
    ai_output_validity: float,
    ai_confidence: float,
    ai_agreement: float,
    human_verification: float,
    human_confirmation_bias: float,
    human_ai_reliance: float,
    human_knowledge: float,
    human_creativity_preference: float,
    rng: np.random.Generator,
    forced_action: HumanAction | None = None,
) -> EvaluationResult:
    # Convert AI output characteristics and human traits into an action.
    # detection: whether the human notices an AI error;
    # acceptance: whether the human adopts the response;
    # verification: whether the human checks it;
    # challenge/exploration: whether the human seeks an alternative.
    #`forced_action` is intended for deterministic unit tests and scenario checks.
    values = {
        "human_input_validity": human_input_validity,
        "ai_output_validity": ai_output_validity,
        "ai_confidence": ai_confidence,
        "ai_agreement": ai_agreement,
        "human_verification": human_verification,
        "human_confirmation_bias": human_confirmation_bias,
        "human_ai_reliance": human_ai_reliance,
        "human_knowledge": human_knowledge,
        "human_creativity_preference": human_creativity_preference,
    }
    for name, value in values.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1; got {value}.")

    ai_error = 1.0 - ai_output_validity
    calibration_gap = max(0.0, ai_confidence - ai_output_validity)

    # Error detection rises with verification and knowledge, but falls when
    # confidence, agreement, confirmation bias, and reliance mask the error.
    detection_probability = _clip(
        0.50 * human_verification
        + 0.35 * human_knowledge
        + 0.25 * ai_error
        - 0.20 * calibration_gap
        - 0.15 * ai_agreement * human_confirmation_bias
        - 0.15 * human_ai_reliance
    )
    detected_error = bool(rng.random() < detection_probability)

    acceptance_probability = _clip(
        0.35 * ai_output_validity
        + 0.20 * ai_confidence
        + 0.20 * ai_agreement
        + 0.15 * human_ai_reliance
        + 0.10 * human_confirmation_bias
        - 0.35 * float(detected_error)
    )

    verification_probability = _clip(
        0.55 * human_verification
        + 0.20 * human_knowledge
        + 0.20 * ai_error
        + 0.10 * calibration_gap
        - 0.15 * human_ai_reliance
    )

    challenge_probability = _clip(
        0.35 * human_creativity_preference
        + 0.25 * human_knowledge
        + 0.20 * float(detected_error)
        + 0.10 * abs(human_input_validity - ai_output_validity)
        - 0.10 * human_confirmation_bias
    )

    perceived_ai_quality = _clip(
        0.45 * ai_output_validity
        + 0.25 * ai_confidence
        + 0.20 * ai_agreement
        - 0.20 * calibration_gap
    )

    if forced_action is not None:
        action = forced_action
    elif detected_error and verification_probability >= 0.50:
        action = "correct"
    else:
        draw = rng.random()
        p_verify = verification_probability
        p_challenge = challenge_probability * (1.0 - p_verify)
        p_accept = acceptance_probability * (1.0 - p_verify - p_challenge)
        p_explore = (
            human_creativity_preference
            * 0.25
            * max(0.0, 1.0 - p_verify - p_challenge - p_accept)
        )
        cumulative = p_verify
        if draw < cumulative:
            action = "verify"
        else:
            cumulative += p_challenge
            if draw < cumulative:
                action = "challenge"
            else:
                cumulative += p_accept
                if draw < cumulative:
                    action = "accept"
                else:
                    cumulative += p_explore
                    action = "explore" if draw < cumulative else "reject"

    return EvaluationResult(
        action=action,
        detected_error=detected_error,
        acceptance_probability=acceptance_probability,
        verification_probability=verification_probability,
        challenge_probability=challenge_probability,
        perceived_ai_quality=perceived_ai_quality,
    )
