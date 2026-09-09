from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np
from .model import AIState, PlatformState, RecursiveModelConfig

def _clip(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))

@dataclass(frozen=True)
class PlatformUpdateResult:
    updated: bool
    update_reason: str
    sample_size: int
    eligible_event_count: int
    sample_validity: float | None
    sample_novelty: float | None
    sample_error: float | None
    sample_ai_origin_share: float | None
    model_version_before: str
    model_version_after: str

def should_update_platform(*, time: int, learning_enabled: bool, update_interval: int) -> bool:
    if update_interval <= 0:
        raise ValueError("update_interval must be positive.")
    return bool(learning_enabled and (time + 1) % update_interval == 0)

def sample_training_events(*, events: list[dict[str, Any]], sampling_rate: float, rng: np.random.Generator) -> list[dict[str, Any]]:
    if not 0.0 <= sampling_rate <= 1.0:
        raise ValueError("sampling_rate must be in [0,1].")
    eligible = [
        e for e in events
        if e.get("event_type") == "interaction_cycle"
        and not e.get("is_intervention", False)
    ]
    if not eligible or sampling_rate == 0.0:
        return []
    n = min(len(eligible), max(1, int(round(len(eligible) * sampling_rate))))
    idx = rng.choice(len(eligible), size=n, replace=False)
    return [eligible[int(i)] for i in np.atleast_1d(idx)]

def update_platform_state(
    *,
    platform: PlatformState,
    global_ai: AIState,
    events: list[dict[str, Any]],
    config: RecursiveModelConfig,
    rng: np.random.Generator,
    force_update: bool = False,
    verified_training_share: float = 0.0,
    human_training_share: float = 0.0,
    contaminated_training_share: float | None = None,
) -> PlatformUpdateResult:
    before = global_ai.model_version
    enabled = bool(config.platform_learning_enabled or force_update)
    if not enabled:
        return PlatformUpdateResult(False, "platform_learning_disabled", 0, 0, None, None, None, None, before, before)

    sampled = sample_training_events(
        events=events,
        sampling_rate=config.platform_interaction_sampling_rate,
        rng=rng,
    )
    eligible_count = sum(
        e.get("event_type") == "interaction_cycle"
        and not e.get("is_intervention", False)
        for e in events
    )
    if not sampled:
        return PlatformUpdateResult(False, "no_sampled_interactions", 0, eligible_count, None, None, None, None, before, before)

    sample_validity = float(np.mean([e.get("final_validity", 0.0) for e in sampled]))
    sample_novelty = float(np.mean([e.get("final_novelty", 0.0) for e in sampled]))
    sample_error = float(np.mean([e.get("error_severity", 0.0) for e in sampled]))
    sample_ai_share = float(np.mean([e.get("ai_origin_share", 0.0) for e in sampled]))
    correction_rate = float(np.mean([bool(e.get("error_corrected", False)) for e in sampled]))

    contaminated = sample_error * sample_ai_share if contaminated_training_share is None else _clip(contaminated_training_share)
    verified = _clip(verified_training_share)
    human = _clip(human_training_share)
    lr = _clip(config.platform_learning_rate)

    quality_signal = _clip(0.50*sample_validity + 0.30*verified + 0.20*human)
    contamination_signal = _clip(0.60*contaminated + 0.40*sample_error*sample_ai_share)

    platform.model_contamination = _clip(
        (1-lr)*platform.model_contamination + lr*contamination_signal - lr*0.50*verified
    )
    platform.response_diversity = _clip(
        (1-lr)*platform.response_diversity + lr*sample_novelty + lr*0.20*human - lr*0.20*sample_ai_share
    )
    platform.performance = _clip(
        (1-lr)*platform.performance + lr*quality_signal - lr*platform.model_contamination
    )

    global_ai.accuracy = _clip(
        (1-lr)*global_ai.accuracy + lr*quality_signal - lr*platform.model_contamination
    )
    global_ai.novelty_capacity = _clip(
        (1-lr)*global_ai.novelty_capacity + lr*platform.response_diversity
    )
    global_ai.model_contamination = platform.model_contamination
    global_ai.response_diversity = platform.response_diversity
    global_ai.sycophancy = _clip(
        global_ai.sycophancy
        + lr*0.20*sample_ai_share*sample_error
        - lr*0.25*verified
        - lr*0.15*correction_rate
    )

    platform.generation += 1
    platform.update_count += 1
    platform.model_version = f"{config.model_id}_G{platform.generation:03d}"
    global_ai.model_version = platform.model_version

    return PlatformUpdateResult(
        True,
        "scheduled_or_forced_platform_update",
        len(sampled),
        eligible_count,
        sample_validity,
        sample_novelty,
        sample_error,
        sample_ai_share,
        before,
        global_ai.model_version,
    )
