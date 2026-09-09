import pandas as pd

from src.metrics.pipeline import MetricsPipelineConfig, compute_run_metrics


def make_frames():
    events = pd.DataFrame([
        {
            "run_id": "R1", "scenario_id": "S0", "time": t,
            "event_type": "interaction_cycle", "is_intervention": False,
            "human_input_validity": 0.7, "ai_output_validity": 0.8,
            "final_validity": 0.75 + 0.01*t,
            "final_novelty": 0.5 + 0.01*t,
            "alignment": 0.9,
            "error_severity": 0.2,
            "repetition_score": 0.3,
            "distortion_score": 0.1,
            "ai_origin_share": 0.5,
            "ai_confidence": 0.8,
            "co_thinking": 0.6,
            "co_movement": 0.05,
            "source_independence": 0.5,
        }
        for t in range(5)
    ] + [{
        "run_id": "R1", "scenario_id": "S0", "time": 2,
        "event_type": "critic_feedback", "is_intervention": True,
    }])

    ideas = pd.DataFrame([
        {
            "run_id": "R1", "scenario_id": "S0", "time": t,
            "idea_id": f"I{t}", "parent_idea_id": f"I{t-1}" if t else None,
            "lineage_id": "L1", "validity": 0.75 + 0.01*t,
            "novelty": 0.5 + 0.01*t, "alignment": 0.9,
            "error_severity": 0.2, "alive": True,
            "descendant_count": 1 if t < 4 else 0,
        }
        for t in range(5)
    ])

    human = pd.DataFrame([
        {
            "run_id": "R1", "scenario_id": "S0", "time": t,
            "state_phase": "post", "verification": 0.6,
            "ai_reliance": 0.5, "independent_performance": 0.7,
        }
        for t in range(5)
    ])
    ai = pd.DataFrame([
        {
            "run_id": "R1", "scenario_id": "S0", "time": t,
            "state_phase": "post", "context_contamination": 0.1,
        }
        for t in range(5)
    ])
    platform = pd.DataFrame([
        {
            "run_id": "R1", "scenario_id": "S0", "time": t,
            "model_contamination": 0.05,
            "platform_performance": 0.8,
        }
        for t in range(5)
    ])
    return events, ideas, human, ai, platform


def test_pipeline_ignores_intervention_events():
    events, ideas, human, ai, platform = make_frames()
    result = compute_run_metrics(
        events=events,
        idea_states=ideas,
        human_states=human,
        ai_states=ai,
        platform_states=platform,
        config=MetricsPipelineConfig(window_size=3),
    )
    assert len(result) == 5
    assert result["window_size"].max() == 3


def test_pipeline_emits_required_metrics():
    events, ideas, human, ai, platform = make_frames()
    result = compute_run_metrics(
        events=events,
        idea_states=ideas,
        human_states=human,
        ai_states=ai,
        platform_states=platform,
    )
    required = {
        "co_thinking",
        "quality_adjusted_comovement",
        "lineage_concentration",
        "effective_lineage_diversity",
        "valid_idea_survival",
        "tail_idea_survival",
        "R_V",
        "R_N",
        "R_E",
        "NER",
        "co_development_score",
        "co_degradation_score",
        "recovery_probability",
        "co_extinction_risk",
        "recursive_degradation_risk",
        "intervention_urgency",
    }
    assert required.issubset(result.columns)
