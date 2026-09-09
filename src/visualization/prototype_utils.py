from __future__ import annotations

from dataclasses import fields, is_dataclass, replace
from pathlib import Path
from typing import Any

import pandas as pd

from src.environment import RecursiveModelConfig, run_recursive_interaction
from src.metrics.pipeline import MetricsPipelineConfig, compute_run_metrics
from src.scenarios import ScenarioRuntime, load_scenario


SCENARIO_OPTIONS = {
    "S0 No treatment": "config/scenarios/S0_no_treatment.yaml",
    "S1 Human data injection": "config/scenarios/S1_human_data_injection.yaml",
    "S2 Critic feedback": "config/scenarios/S2_critic_feedback.yaml",
    "S3 Context reset": "config/scenarios/S3_context_reset.yaml",
}


def available_config_fields() -> set[str]:
    if is_dataclass(RecursiveModelConfig):
        return {f.name for f in fields(RecursiveModelConfig)}
    return set()


def safe_config_kwargs(candidate: dict[str, Any]) -> dict[str, Any]:
    valid = available_config_fields()
    return {k: v for k, v in candidate.items() if k in valid}


def replace_nested_attr_if_exists(obj: Any, dotted_path: str, value: Any) -> Any:

    if value is None:
        return obj

    parts = dotted_path.split(".")
    if not parts:
        return obj

    def _replace(current: Any, remaining: list[str]) -> tuple[Any, bool]:
        name = remaining[0]

        if not hasattr(current, name):
            return current, False

        if len(remaining) == 1:
            if is_dataclass(current):
                return replace(current, **{name: value}), True

            try:
                setattr(current, name, value)
                return current, True
            except Exception:
                return current, False

        child = getattr(current, name)
        new_child, changed = _replace(child, remaining[1:])
        if not changed:
            return current, False

        if is_dataclass(current):
            return replace(current, **{name: new_child}), True

        try:
            setattr(current, name, new_child)
            return current, True
        except Exception:
            return current, False

    new_obj, _ = _replace(obj, parts)
    return new_obj


# Backward-compatible alias used by the initial Stage 11 app.
def set_nested_attr_if_exists(obj: Any, dotted_path: str, value: Any) -> bool:
    return replace_nested_attr_if_exists(obj, dotted_path, value) is not obj


def scenario_id_from_path(path: str | Path) -> str:
    scenario = load_scenario(str(path))
    return scenario.id


def build_config(
    *,
    scenario_id: str,
    run_id: str,
    steps: int,
    seed: int,
    controls: dict[str, Any],
) -> RecursiveModelConfig:
   
    candidate = {
        "num_steps": steps,
        "seed": seed,
        "run_id": run_id,
        "scenario_id": scenario_id,
        "human_accuracy": controls.get("human_accuracy"),
        "ai_accuracy": controls.get("ai_accuracy"),
        "verification": controls.get("verification"),
        "human_verification": controls.get("verification"),
        "sycophancy": controls.get("sycophancy"),
        "confidence_calibration": controls.get("confidence_calibration"),
        "ai_feedback_reuse": controls.get("ai_reuse_ratio"),
        "mutation_rate": controls.get("mutation_rate"),
        "platform_update_interval": controls.get("retraining_interval"),
        "platform_learning_enabled": controls.get("platform_learning_enabled"),
        "platform_interaction_sampling_rate": controls.get(
            "platform_interaction_sampling_rate"
        ),
        "platform_learning_rate": controls.get("platform_learning_rate"),
    }

    kwargs = safe_config_kwargs({k: v for k, v in candidate.items() if v is not None})
    return RecursiveModelConfig(**kwargs)


def load_scenario_with_overrides(
    scenario_path: str | Path,
    controls: dict[str, Any],
):

    scenario = load_scenario(str(scenario_path))

    overrides = [
        ("interaction.ai_reuse_ratio", controls.get("ai_reuse_ratio")),
        ("interaction.context_reset_strength", controls.get("context_reset_strength")),
        ("interaction.preserve_verified_summary", controls.get("preserve_verified_summary")),
        ("interaction.trigger.threshold_value", controls.get("reset_threshold")),
        ("interaction.trigger.cooldown", controls.get("reset_cooldown")),
        ("interaction.trigger.start_time", controls.get("intervention_start_time")),
        ("interaction.trigger.interval", controls.get("intervention_interval")),
        ("interaction.trigger.time", controls.get("human_data_injection_time")),
        ("platform.update_interval", controls.get("retraining_interval")),
        ("platform.learning_enabled", controls.get("platform_learning_enabled")),
        ("platform.learning_rate", controls.get("platform_learning_rate")),
        (
            "platform.interaction_sampling_rate",
            controls.get("platform_interaction_sampling_rate"),
        ),
    ]

    for path, value in overrides:
        scenario = replace_nested_attr_if_exists(scenario, path, value)

    return scenario


def run_prototype_simulation(
    *,
    scenario_path: str | Path,
    steps: int,
    seed: int,
    controls: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    scenario = load_scenario_with_overrides(scenario_path, controls)
    run_id = f"{scenario.id}_prototype"

    config = build_config(
        scenario_id=scenario.id,
        run_id=run_id,
        steps=steps,
        seed=seed,
        controls=controls,
    )

    result = run_recursive_interaction(
        config,
        scenario_runtime=ScenarioRuntime(scenario),
    )
    frames = result.as_frames()

    metrics = compute_metrics_from_frames(frames)
    frames["outcome_metrics"] = metrics
    return frames


def compute_metrics_from_frames(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    required = [
        "events",
        "idea_states",
        "human_states",
        "ai_states",
        "platform_states",
    ]
    if any(name not in frames or frames[name].empty for name in required):
        return pd.DataFrame()

    try:
        return compute_run_metrics(
            events=frames["events"],
            idea_states=frames["idea_states"],
            human_states=frames["human_states"],
            ai_states=frames["ai_states"],
            platform_states=frames["platform_states"],
            config=MetricsPipelineConfig(show_progress=False),
        )
    except TypeError:
        return compute_run_metrics(
            events=frames["events"],
            idea_states=frames["idea_states"],
            human_states=frames["human_states"],
            ai_states=frames["ai_states"],
            platform_states=frames["platform_states"],
        )


def current_regime(metrics: pd.DataFrame) -> str:

    if metrics.empty:
        return "Unavailable"

    latest = metrics.sort_values("time").iloc[-1]
    development = float(latest.get("co_development_score", 0.0))
    risk = float(latest.get("recursive_degradation_risk", 0.0))

    if development >= 0.50 and risk < 0.20:
        return "Productive co-development"
    if development >= 0.50 and risk >= 0.20:
        return "Productive but fragile"
    if development < 0.50 and risk >= 0.20:
        return "Recursive degradation"
    return "Stable low-activity"


def intervention_recommendation(metrics: pd.DataFrame, local_ai_states: pd.DataFrame) -> str:

    if metrics.empty:
        return "Run a simulation to generate a recommendation."

    latest = metrics.sort_values("time").iloc[-1]
    risk = float(latest.get("recursive_degradation_risk", 0.0))
    recovery = float(latest.get("recovery_probability", 0.0))
    error_reproduction = float(latest.get("R_E", 0.0))

    context_contamination = 0.0
    if local_ai_states is not None and not local_ai_states.empty:
        if "context_contamination" in local_ai_states.columns:
            context_contamination = float(
                local_ai_states.sort_values("time")["context_contamination"].iloc[-1]
            )

    if context_contamination >= 0.02:
        return "Consider context reset: local context contamination is elevated."
    if error_reproduction >= 0.30:
        return "Consider verification or human-origin data: error-bearing activity is elevated."
    if risk >= 0.20 and recovery < 0.65:
        return "Consider critic feedback with verification: degradation risk is elevated."
    return "No urgent intervention indicated under the current thresholds."


def summarize_latest(metrics: pd.DataFrame) -> dict[str, float]:
    if metrics.empty:
        return {}

    latest = metrics.sort_values("time").iloc[-1]
    keys = [
        "co_development_score",
        "co_degradation_score",
        "co_extinction_risk",
        "recursive_degradation_risk",
        "R_V",
        "R_E",
        "NER",
        "recovery_probability",
        "mean_validity",
        "mean_novelty",
        "mean_alignment",
    ]
    return {
        key: float(latest[key])
        for key in keys
        if key in latest.index and pd.notna(latest[key])
    }


def export_frames_to_directory(
    frames: dict[str, pd.DataFrame],
    output_dir: str | Path,
) -> list[Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    saved = []
    for name, frame in frames.items():
        if isinstance(frame, pd.DataFrame) and not frame.empty:
            path = out / f"{name}.csv"
            frame.to_csv(path, index=False)
            saved.append(path)
    return saved
