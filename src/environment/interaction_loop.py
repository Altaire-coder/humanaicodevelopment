from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from uuid import uuid4

import numpy as np

from .interaction_state import (
    LocalAIState,
    update_interaction_state,
)
from .platform_loop import (
    should_update_platform,
    update_platform_state,
)
from .platform_history import PlatformHistory

from src.interactions import InteractionConfig, run_interaction
from src.interventions import (
    apply_context_reset,
    apply_critic_feedback,
    apply_human_data_injection,
    )
from src.metrics.pipeline import MetricsPipelineConfig
from src.metrics.runtime import compute_runtime_metrics
from src.scenarios import ScenarioRuntime

from .model import (
    AIState,
    HumanState,
    PlatformState,
    RecursiveModelConfig,
    RecursiveSimulationResult,
)


_FAST_TRIGGER_METRICS = {"repetition_rate", "context_contamination"}


def _clip(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _scenario_needs_full_runtime_metrics(scenario_runtime: ScenarioRuntime | None) -> bool:
    if scenario_runtime is None:
        return False
    scenario = scenario_runtime.scenario
    if scenario.diagnosis_enabled:
        return True
    triggers = [scenario.interaction.trigger, scenario.platform.trigger]
    for trigger in triggers:
        if trigger.type == "predictive":
            return True
        if trigger.type == "threshold" and trigger.threshold_metric not in _FAST_TRIGGER_METRICS:
            return True
    return False


def _fast_runtime_metrics(
    *,
    output: RecursiveSimulationResult,
    human: HumanState,
    local_ai: LocalAIState,
    platform: PlatformState,
    window_size: int = 10,
) -> dict[str, float]:
    interactions = [
        event
        for event in output.events
        if event.get("event_type") == "interaction_cycle"
        and not event.get("is_intervention", False)
    ]
    window = interactions[-window_size:]
    repetition_values = [
        float(event.get("repetition_score", 0.0) or 0.0)
        for event in window
    ]
    repetition = (
        float(np.mean([value >= 0.80 for value in repetition_values]))
        if repetition_values
        else 0.0
    )
    mean_validity = (
        float(np.mean([event.get("final_validity", 0.0) for event in window]))
        if window
        else 0.0
    )
    performance_decline = max(0.0, 0.75 - mean_validity) if window else 0.0
    return {
        "repetition_rate": repetition,
        "context_contamination": float(local_ai.context_contamination),
        "model_contamination": float(platform.model_contamination),
        "lineage_concentration": 0.0,
        "effective_lineage_diversity": 1.0,
        "diversity_loss": 0.0,
        "R_V": 0.0,
        "R_N": 0.0,
        "R_E": 0.0,
        "NER": 0.0,
        "recovery_probability": 1.0,
        "co_development_score": mean_validity,
        "co_degradation_score": performance_decline,
        "co_extinction_risk": performance_decline,
        "recursive_degradation_risk": performance_decline,
        "intervention_urgency": 0.0,
        "predicted_failure_probability": performance_decline,
        "cross_session_error_rate": float(platform.model_contamination),
        "utility_quality_gap": max(0.0, mean_validity - float(platform.performance)),
    }


def _runtime_metrics(
    *,
    output: RecursiveSimulationResult,
    human: HumanState,
    local_ai: LocalAIState,
    platform: PlatformState,
    scenario_runtime: ScenarioRuntime | None,
) -> dict[str, float]:
    if _scenario_needs_full_runtime_metrics(scenario_runtime):
        return compute_runtime_metrics(
            output=output,
            config=MetricsPipelineConfig(
                window_size=10,
                baseline_window=5,
            ),
        )
    return _fast_runtime_metrics(
        output=output,
        human=human,
        local_ai=local_ai,
        platform=platform,
    )


def _generate_human_input(
    human: HumanState,
    previous_idea: dict,
    config: RecursiveModelConfig,
    rng: np.random.Generator,
) -> dict:
    ai_share = _clip(config.ai_feedback_reuse * human.ai_reliance)
    human_share = _clip(1.0 - ai_share)

    independent_validity = _clip(
        rng.normal(
            0.55 * human.knowledge + 0.45 * human.independent_performance,
            0.08,
        )
    )
    independent_novelty = _clip(
        rng.normal(human.creativity_preference, 0.10)
    )

    validity = _clip(
        human_share * independent_validity
        + ai_share * previous_idea["validity"]
        - human.error_internalization * 0.10
    )
    novelty = _clip(
        human_share * independent_novelty
        + ai_share * previous_idea["novelty"]
        - ai_share * previous_idea.get("repetition_score", 0.0) * 0.10
    )
    alignment = _clip(
        human_share * human.alignment
        + ai_share * previous_idea["alignment"]
    )
    confidence = _clip(
        0.55 * human.confidence
        + 0.45 * validity
        + rng.normal(0.0, 0.03)
    )

    return {
        "validity": validity,
        "novelty": novelty,
        "alignment": alignment,
        "confidence": confidence,
        "human_origin_share": human_share,
        "ai_origin_share": ai_share,
        "external_origin_share": 0.0,
        "source_independence": human_share,
    }


def _generate_ai_response(
    global_ai: AIState,
    local_ai: LocalAIState,
    human_input: dict,
    previous_idea: dict,
    rng: np.random.Generator,
) -> dict:
    memory_weight = _clip(
        local_ai.memory_strength
    )

    base_accuracy = (
        global_ai.accuracy
        * (
            1.0
            - local_ai.context_contamination
        )
    )
    
    validity = _clip(
        rng.normal(
            0.55 * base_accuracy
            + 0.25 * human_input["validity"]
            + 0.20 * previous_idea["validity"] * memory_weight,
            0.08,
        )
    )
    novelty = _clip(
        rng.normal(
            0.55 * global_ai.novelty_capacity * global_ai.creativity_setting
            + 0.25 * human_input["novelty"]
            + 0.20 * previous_idea["novelty"] * memory_weight,
            0.09,
        )
    )
    alignment = _clip(
        0.50 * global_ai.alignment
        + 0.30 * human_input["alignment"]
        + 0.20 * previous_idea["alignment"] * memory_weight
        - 0.20 * local_ai.context_contamination
    )
    agreement = _clip(
        global_ai.sycophancy
        + global_ai.personalization * 0.40
        + human_input["validity"] * 0.20
    )
    confidence = _clip(
        validity * global_ai.confidence_calibration
        + (1.0 - global_ai.confidence_calibration)
        + rng.normal(0.0, 0.02)
    )
    return {
        "validity": validity,
        "novelty": novelty,
        "alignment": alignment,
        "agreement": agreement,
        "confidence": confidence,
    }




def _log_intervention_event(
    *,
    output: RecursiveSimulationResult,
    decision,
    config: RecursiveModelConfig,
    time: int,
    model_version_before: str,
    model_version_after: str,
) -> None:
    if not decision.triggered:
        return

    output.events.append({
        "run_id": config.run_id,
        "scenario_id": config.scenario_id,
        "event_id": f"EVT_{uuid4().hex[:12]}",
        "interaction_id": f"INTV_{uuid4().hex[:12]}",
        "time": time,
        "substep": 0,
        "generation": time,
        "human_id": config.human_id,
        "ai_id": config.ai_id,
        "platform_id": config.platform_id,
        "model_id": config.model_id,
        "source_id": config.platform_id,
        "source_type": "platform",
        "target_id": config.ai_id,
        "target_type": "ai",
        "event_type": (
            decision.interaction_action
            if decision.interaction_action != "none"
            else decision.platform_action
        ),
        "event_subtype": decision.diagnosis,
        "action": "intervene",
        "is_intervention": True,
        "treatment_id": decision.treatment_id,
        "treatment_level": decision.treatment_level,
        "reason": decision.reason,
        "model_version_before": model_version_before,
        "model_version_after": model_version_after,
        **decision.parameters,
    })


def run_recursive_interaction(
    config: RecursiveModelConfig,
    scenario_runtime: ScenarioRuntime | None = None,
) -> RecursiveSimulationResult:
    rng = np.random.default_rng(config.seed)
    output = RecursiveSimulationResult()

    human = HumanState(
        validity=config.initial_validity,
        novelty=config.initial_novelty,
        alignment=config.initial_alignment,
        confidence=config.initial_human_confidence,
        knowledge=config.human_knowledge,
        verification=config.human_verification,
        confirmation_bias=config.human_confirmation_bias,
        ai_reliance=config.human_ai_reliance,
        creativity_preference=config.human_creativity_preference,
        rigor_preference=config.human_rigor_preference,
        independent_performance=config.initial_validity,
    )
    ai = AIState(
        accuracy=config.ai_accuracy,
        novelty_capacity=config.ai_novelty_capacity,
        alignment=config.ai_alignment,
        confidence_calibration=config.ai_confidence_calibration,
        sycophancy=config.ai_sycophancy,
        memory_strength=config.ai_memory_strength,
        personalization=config.ai_personalization,
        creativity_setting=config.ai_creativity_setting,
        rigor_setting=config.ai_rigor_setting,
        model_version=config.model_id,
    )
    platform = PlatformState(model_version=config.model_id)

    baseline_ai = deepcopy(ai)
    baseline_platform = deepcopy(platform)

    local_ai = LocalAIState(
        context_contamination=0.0,
        memory_strength=config.ai_memory_strength,
        local_response_diversity=1.0,
        local_alignment_conditioning=(
            config.initial_alignment
        ),
        local_error_correction_tendency=0.50,
        current_intention_anchor=(
            config.initial_alignment
        ),
    )

    platform_history = PlatformHistory()

    platform_history.save(
        time=-1,
        platform=platform,
        global_ai=ai,
        reason="initial_state",
    )


    parent_idea_id = "I_000001"
    previous_idea = {
        "idea_id": parent_idea_id,
        "validity": config.initial_validity,
        "novelty": config.initial_novelty,
        "alignment": config.initial_alignment,
        "utility": config.initial_validity * config.initial_alignment,
        "repetition_score": 0.0,
        "error_severity": 1.0 - config.initial_validity,
        "alive": True,
    }

    for t in range(config.num_steps):
        output.human_states.append({
            "run_id": config.run_id,
            "scenario_id": config.scenario_id,
            "time": t,
            "state_phase": "pre",
            "human_id": config.human_id,
            "knowledge": human.knowledge,
            "accuracy": human.validity,
            "confidence": human.confidence,
            "verification": human.verification,
            "ai_reliance": human.ai_reliance,
            "independent_performance": human.independent_performance,
            "error_internalization": human.error_internalization,
        })
        output.ai_states.append({
            "run_id": config.run_id,
            "scenario_id": config.scenario_id,
            "time": t,
            "state_phase": "pre",
            "ai_id": config.ai_id,
            "model_version": ai.model_version,
            "accuracy": ai.accuracy,
            "context_contamination": local_ai.context_contamination,
            "model_contamination": ai.model_contamination,
            "response_diversity": ai.response_diversity,
        })

        # Scenario decisions use only completed history through t-1.
        metrics_before = _runtime_metrics(
            output=output,
            human=human,
            local_ai=local_ai,
            platform=platform,
            scenario_runtime=scenario_runtime,
        )
        decision = (
            scenario_runtime.decide(time=t, metrics=metrics_before)
            if scenario_runtime is not None
            else None
        )

        model_version_before = ai.model_version

        if decision and decision.interaction_action == "context_reset":
            apply_context_reset(
                ai_state=local_ai,
                reset_strength=scenario_runtime.scenario.interaction.context_reset_strength,
                preserve_verified_summary=scenario_runtime.scenario.interaction.preserve_verified_summary,
            )

        human_input = _generate_human_input(
            human, previous_idea, config, rng
        )

        if decision and decision.interaction_action == "human_data_injection":
            policy = scenario_runtime.scenario.interaction
            human_input = apply_human_data_injection(
                human_input,
                injection_share=policy.human_data_injection,
                injected_validity=policy.injected_data_validity,
                injected_novelty=policy.injected_data_novelty,
                injected_alignment=policy.injected_data_alignment,
                source_independence=policy.source_independence,
            )

        ai_output = _generate_ai_response(
            ai,
            local_ai,
            human_input,
            previous_idea,
            rng,
        )

        if decision and decision.interaction_action == "critic_feedback":
            policy = scenario_runtime.scenario.interaction
            ai_output = apply_critic_feedback(
                ai_output,
                critic_strength=policy.critic_strength,
                counterfactual_strength=policy.counterfactual_strength,
                uncertainty_flagging=policy.uncertainty_flagging,
                intention_anchor_strength=policy.intention_anchor_strength,
            )

        interaction = run_interaction(
            human_input_validity=human_input["validity"],
            human_input_novelty=human_input["novelty"],
            human_input_alignment=human_input["alignment"],
            ai_output_validity=ai_output["validity"],
            ai_output_novelty=ai_output["novelty"],
            ai_output_alignment=ai_output["alignment"],
            ai_confidence=ai_output["confidence"],
            config=InteractionConfig(
                accuracy_threshold=config.accuracy_threshold,
                human_knowledge=human.knowledge,
                human_verification=human.verification,
                human_confirmation_bias=human.confirmation_bias,
                human_ai_reliance=human.ai_reliance,
                human_creativity_preference=human.creativity_preference,
                human_rigor_preference=human.rigor_preference,
                ai_agreement=ai_output["agreement"],
            ),
            rng=rng,
            run_id=config.run_id,
            scenario_id=config.scenario_id,
            time=t,
            generation=t + 1,
            human_id=config.human_id,
            ai_id=config.ai_id,
            parent_idea_id=parent_idea_id,
            lineage_id=config.lineage_id,
        )

        event = interaction.to_record()
        event.update({
            "platform_id": config.platform_id,
            "model_id": config.model_id,
            "event_type": "interaction_cycle",
            "source_id": config.human_id,
            "source_type": "human",
            "target_id": config.ai_id,
            "target_type": "ai",
            "human_confidence": human_input["confidence"],
            "verification": human.verification,
            "human_input_origin_share": human_input["human_origin_share"],
            "recursive_ai_input_share": human_input["ai_origin_share"],
            "source_independence": human_input.get("source_independence", 0.0),
            "co_thinking": interaction.co_thinking_score,
            "co_movement": interaction.co_movement_score,
            "is_intervention": False,
        })
        output.events.append(event)

        output.idea_states.append({
            "run_id": config.run_id,
            "scenario_id": config.scenario_id,
            "time": t,
            "idea_id": interaction.new_idea_id,
            "parent_idea_id": interaction.parent_idea_id,
            "lineage_id": interaction.lineage_id,
            "generation": interaction.generation,
            "origin_type": "hybrid",
            "validity": interaction.final_validity,
            "novelty": interaction.final_novelty,
            "alignment": interaction.alignment,
            "utility": interaction.utility,
            "confidence": interaction.final_confidence,
            "creativity": interaction.creativity,
            "technical_rigor": interaction.technical_rigor,
            "repetition_score": interaction.repetition_score,
            "distortion_score": interaction.distortion_score,
            "error_severity": interaction.error_severity,
            "mutation_type": interaction.mutation_type,
            "human_origin_share": interaction.human_origin_share,
            "ai_origin_share": interaction.ai_origin_share,
            "source_independence": human_input.get("source_independence", 0.0),
            "alive": interaction.alive,
            "survival_duration": 1,
            "descendant_count": 0,
        })
        output.lineage_edges.append({
            "run_id": config.run_id,
            "scenario_id": config.scenario_id,
            "time": t,
            "edge_id": f"LE_{uuid4().hex[:10]}",
            "source_id": interaction.parent_idea_id,
            "target_id": interaction.new_idea_id,
            "edge_type": "idea_inheritance",
            "idea_id": interaction.new_idea_id,
            "lineage_id": interaction.lineage_id,
            "event_id": interaction.event_id,
            "validity": interaction.final_validity,
            "mutation_type": interaction.mutation_type,
            "is_intervention": False,
        })

        interaction_update = update_interaction_state(
            human=human,
            local_ai=local_ai,
            interaction=interaction,
            config=config,
        )
        
        platform_update = None

        # Scenario-level platform interventions take precedence over scheduled learning.
        if (
            decision
            and decision.platform_action == "verified_retraining"
        ):
            policy = scenario_runtime.scenario.platform
            platform_update = update_platform_state(
                platform=platform,
                global_ai=ai,
                events=output.events,
                config=config,
                rng=rng,
                force_update=True,
                verified_training_share=policy.verified_training_share,
                human_training_share=policy.human_training_share,
                contaminated_training_share=policy.contaminated_training_share,
            )
            if platform_update.updated:
                platform_history.save(
                    time=t,
                    platform=platform,
                    global_ai=ai,
                    reason="verified_retraining",
                )

        elif decision and decision.platform_action == "model_rollback":
            platform, ai = platform_history.rollback(
                steps=scenario_runtime.scenario.platform.rollback_steps
            )

        elif decision and decision.platform_action == "model_reset":
            strength = scenario_runtime.scenario.platform.reset_strength
            platform.model_contamination = (
                (1 - strength) * platform.model_contamination
                + strength * baseline_platform.model_contamination
            )
            platform.response_diversity = (
                (1 - strength) * platform.response_diversity
                + strength * baseline_platform.response_diversity
            )
            platform.performance = (
                (1 - strength) * platform.performance
                + strength * baseline_platform.performance
            )
            ai.accuracy = (
                (1 - strength) * ai.accuracy
                + strength * baseline_ai.accuracy
            )
            ai.model_contamination = (
                (1 - strength) * ai.model_contamination
                + strength * baseline_ai.model_contamination
            )
            ai.response_diversity = (
                (1 - strength) * ai.response_diversity
                + strength * baseline_ai.response_diversity
            )
            ai.sycophancy = (
                (1 - strength) * ai.sycophancy
                + strength * baseline_ai.sycophancy
            )
            ai.model_version = baseline_ai.model_version
            platform.model_version = baseline_platform.model_version

        elif should_update_platform(
            time=t,
            learning_enabled=config.platform_learning_enabled,
            update_interval=config.platform_update_interval,
        ):
            platform_update = update_platform_state(
                platform=platform,
                global_ai=ai,
                events=output.events,
                config=config,
                rng=rng,
            )
            if platform_update.updated:
                platform_history.save(
                    time=t,
                    platform=platform,
                    global_ai=ai,
                    reason="scheduled_platform_learning",
                )

        if platform_update is not None and platform_update.updated:
            output.platform_update_events.append({
                "run_id": config.run_id,
                "scenario_id": config.scenario_id,
                "time": t,
                "event_type": "platform_update",
                "update_reason": platform_update.update_reason,
                "sample_size": platform_update.sample_size,
                "eligible_event_count": platform_update.eligible_event_count,
                "sample_validity": platform_update.sample_validity,
                "sample_novelty": platform_update.sample_novelty,
                "sample_error": platform_update.sample_error,
                "sample_ai_origin_share": platform_update.sample_ai_origin_share,
                "model_version_before": platform_update.model_version_before,
                "model_version_after": platform_update.model_version_after,
            })

        model_version_after = ai.model_version
        if decision:
            _log_intervention_event(
                output=output,
                decision=decision,
                config=config,
                time=t,
                model_version_before=model_version_before,
                model_version_after=model_version_after,
            )

        # Save current post-interaction states before computing current metrics.
        output.human_states.append({
            "run_id": config.run_id,
            "scenario_id": config.scenario_id,
            "time": t,
            "state_phase": "post",
            "human_id": config.human_id,
            "knowledge": human.knowledge,
            "accuracy": human.validity,
            "confidence": human.confidence,
            "verification": human.verification,
            "ai_reliance": human.ai_reliance,
            "independent_performance": human.independent_performance,
            "error_internalization": human.error_internalization,
        })
        output.ai_states.append({
            "run_id": config.run_id,
            "scenario_id": config.scenario_id,
            "time": t,
            "state_phase": "post",
            "ai_id": config.ai_id,
            "model_version": ai.model_version,
            "accuracy": ai.accuracy,
            "novelty_capacity": (
                ai.novelty_capacity
            ),
            "response_diversity": (
                ai.response_diversity
            ),
            "sycophancy": ai.sycophancy,
            "model_contamination": (
                ai.model_contamination
            ),
        })
        output.local_ai_states.append({
            "run_id": config.run_id,
            "scenario_id": config.scenario_id,
            "time": t,
            "state_phase": "post",
            "ai_id": config.ai_id,
            "context_contamination": (
                local_ai.context_contamination
            ),
            "memory_strength": (
                local_ai.memory_strength
            ),
            "local_response_diversity": (
                local_ai.local_response_diversity
            ),
            "local_alignment_conditioning": (
                local_ai.local_alignment_conditioning
            ),
            "local_error_correction_tendency": (
                local_ai
                .local_error_correction_tendency
            ),
            "current_intention_anchor": (
                local_ai.current_intention_anchor
            ),
            "session_interaction_count": (
                local_ai.session_interaction_count
            ),
        })

        # Append the raw platform state first. 
        output.platform_states.append({
            "run_id": config.run_id,
            "scenario_id": config.scenario_id,
            "time": t,
            "platform_id": config.platform_id,
            "model_version": platform.model_version,
            "active_humans": 1,
            "active_ai_agents": 1,
            "active_ideas": len(output.idea_states) + 1,
            "valid_ideas": sum(
                x["validity"] >= config.accuracy_threshold
                for x in output.idea_states
            ),
            "erroneous_ideas": sum(
                x["validity"] < config.accuracy_threshold
                for x in output.idea_states
            ),
            "extinct_ideas": sum(not x["alive"] for x in output.idea_states),
            "platform_performance": platform.performance,
            "context_contamination": local_ai.context_contamination,
            "model_contamination": platform.model_contamination,
            "response_diversity": platform.response_diversity,
            "scenario_triggered": bool(decision and decision.triggered),
            "interaction_action": (
                decision.interaction_action if decision else "none"
            ),
            "platform_action": (
                decision.platform_action if decision else "none"
            ),
            "diagnosis": (
                decision.diagnosis if decision else "not_applicable"
            ),
        })

        metrics_after = _runtime_metrics(
            output=output,
            human=human,
            local_ai=local_ai,
            platform=platform,
            scenario_runtime=scenario_runtime,
        )
        output.platform_states[-1].update(metrics_after)

        previous_idea = {
            "idea_id": interaction.new_idea_id,
            "validity": interaction.final_validity,
            "novelty": interaction.final_novelty,
            "alignment": interaction.alignment,
            "utility": interaction.utility,
            "repetition_score": interaction.repetition_score,
            "error_severity": interaction.error_severity,
            "alive": interaction.alive,
        }
        parent_idea_id = interaction.new_idea_id

    return output
