from __future__ import annotations

import pandas as pd

from .pipeline import MetricsPipelineConfig, compute_run_metrics


def _latest_local_context_contamination(frames: dict[str, pd.DataFrame]) -> float:
    local_ai_states = frames.get("local_ai_states", pd.DataFrame())

    if (
        local_ai_states is None
        or local_ai_states.empty
        or "context_contamination" not in local_ai_states.columns
    ):
        return 0.0

    local = local_ai_states.copy()

    if "state_phase" in local.columns:
        post = local[local["state_phase"].eq("post")]
        if not post.empty:
            local = post

    if "time" in local.columns:
        local = local.sort_values("time")

    value = local.iloc[-1].get("context_contamination", 0.0)
    if pd.isna(value):
        return 0.0

    return float(value)


def compute_runtime_metrics(
    *,
    output,
    config: MetricsPipelineConfig | None = None,
) -> dict[str, float]:

    frames = output.as_frames()
    local_context_contamination = _latest_local_context_contamination(frames)

    metrics = compute_run_metrics(
        events=frames["events"],
        idea_states=frames["idea_states"],
        human_states=frames["human_states"],
        ai_states=frames["ai_states"],
        platform_states=frames["platform_states"],
        config=config,
    )

    if metrics.empty:
        return {
            "repetition_rate": 0.0,
            "context_contamination": local_context_contamination,
            "model_contamination": 0.0,
            "lineage_concentration": 0.0,
            "effective_lineage_diversity": 0.0,
            "diversity_loss": 0.0,
            "R_V": 0.0,
            "R_N": 0.0,
            "R_E": 0.0,
            "NER": 0.0,
            "recovery_probability": 1.0,
            "co_development_score": 0.0,
            "co_degradation_score": 0.0,
            "co_extinction_risk": 0.0,
            "recursive_degradation_risk": 0.0,
            "intervention_urgency": 0.0,
            "predicted_failure_probability": 0.0,
            "cross_session_error_rate": 0.0,
            "utility_quality_gap": 0.0,
        }

    latest = metrics.sort_values("time").iloc[-1].to_dict()

    latest["context_contamination"] = local_context_contamination

    latest["diversity_loss"] = max(
        0.0,
        1.0 - float(latest.get("valid_diversity", 0.0)),
    )
    latest["predicted_failure_probability"] = float(
        latest.get("co_extinction_risk", 0.0)
    )
    latest["cross_session_error_rate"] = float(
        latest.get("model_contamination", 0.0)
    )
    latest["utility_quality_gap"] = max(
        0.0,
        float(latest.get("mean_validity", 0.0))
        - float(latest.get("platform_performance", 0.0)),
    )
    return latest
