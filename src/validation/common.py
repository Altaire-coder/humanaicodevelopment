from __future__ import annotations

from dataclasses import fields, is_dataclass, replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

from src.environment import RecursiveModelConfig, run_recursive_interaction
from src.metrics.pipeline import MetricsPipelineConfig, compute_run_metrics
from src.metrics.accuracy import calibration_error
from src.metrics.development import (
    co_degradation_score,
    co_development_score,
    net_epistemic_reproduction,
    preservation_score,
    productive_expansion_score,
)
from src.metrics.diversity import (
    effective_lineage_diversity,
    lineage_concentration,
    repetition_rate,
    tail_idea_survival,
)
from src.metrics.extinction import co_extinction_risk, recovery_probability
from src.metrics.genealogy import reproduction_numbers
from src.metrics.risk import intervention_urgency, recursive_degradation_risk
from src.scenarios import ScenarioRuntime, load_scenario


DEFAULT_SCENARIOS = {
    "S0_no_treatment": "config/scenarios/S0_no_treatment.yaml",
    "S1_human_data_injection": "config/scenarios/S1_human_data_injection.yaml",
    "S2_critic_feedback": "config/scenarios/S2_critic_feedback.yaml",
    "S3_context_reset": "config/scenarios/S3_context_reset.yaml",
}

OUTPUT_TABLES = (
    "events",
    "idea_states",
    "human_states",
    "ai_states",
    "local_ai_states",
    "platform_states",
    "platform_update_events",
    "lineage_edges",
)

CORE_METRICS = [
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


def progress_iter(iterable: Iterable, *, total: int | None = None, desc: str = "", disable: bool = False):
    if tqdm is None or disable:
        return iterable
    return tqdm(iterable, total=total, desc=desc)


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


def load_scenario_with_overrides(scenario_path: str | Path, controls: dict[str, Any] | None = None):
    controls = controls or {}
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
        ("platform.interaction_sampling_rate", controls.get("platform_interaction_sampling_rate")),
    ]
    for path, value in overrides:
        scenario = replace_nested_attr_if_exists(scenario, path, value)
    return scenario


def build_config(*, scenario_id: str, run_id: str, steps: int, seed: int, controls: dict[str, Any] | None = None) -> RecursiveModelConfig:
    controls = controls or {}
    candidate = {
        "num_steps": steps,
        "seed": seed,
        "run_id": run_id,
        "scenario_id": scenario_id,
        "human_accuracy": controls.get("human_accuracy"),
        "ai_accuracy": controls.get("ai_accuracy"),
        "verification": controls.get("verification"),
        "human_verification": controls.get("verification"),
        "human_learning_rate": controls.get("human_learning_rate"),
        "human_dependence_rate": controls.get("human_dependence_rate"),
        "sycophancy": controls.get("sycophancy"),
        "ai_memory_strength": controls.get("ai_memory_strength"),
        "confidence_calibration": controls.get("confidence_calibration"),
        "ai_feedback_reuse": controls.get("ai_reuse_ratio"),
        "memory_decay": controls.get("memory_decay"),
        "context_contamination_rate": controls.get("context_contamination_rate"),
        "error_persistence": controls.get("error_persistence"),
        "mutation_rate": controls.get("mutation_rate"),
        "platform_update_interval": controls.get("retraining_interval"),
        "platform_learning_enabled": controls.get("platform_learning_enabled"),
        "platform_interaction_sampling_rate": controls.get("platform_interaction_sampling_rate"),
        "platform_learning_rate": controls.get("platform_learning_rate"),
    }
    return RecursiveModelConfig(**safe_config_kwargs({k: v for k, v in candidate.items() if v is not None}))


def compute_metrics_from_frames(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    required = ["events", "idea_states", "human_states", "ai_states", "platform_states"]
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


def _mean(values: Iterable[float], default: float = 0.0) -> float:
    values = list(values)
    return float(np.mean(values)) if values else float(default)


def _state_latest(frame: pd.DataFrame, *, time: int, phase: str | None = None) -> pd.Series | None:
    if frame.empty:
        return None
    subset = frame[frame["time"] <= time]
    if phase is not None and "state_phase" in subset.columns:
        subset = subset[subset["state_phase"].eq(phase)]
    if subset.empty:
        return None
    return subset.sort_values("time").iloc[-1]


def _final_idea_snapshot(ideas: pd.DataFrame, *, time: int) -> pd.DataFrame:
    if ideas.empty:
        return ideas.copy()
    subset = ideas[ideas["time"] <= time].copy()
    if subset.empty:
        return subset
    subset = subset.sort_values(["idea_id", "time"]).groupby("idea_id", as_index=False).tail(1)
    if "alive" in subset.columns:
        subset = subset[subset["alive"].fillna(True)]
    return subset


def _lineage_probabilities(snapshot: pd.DataFrame) -> np.ndarray:
    if snapshot.empty or "lineage_id" not in snapshot.columns:
        return np.asarray([], dtype=float)
    counts = snapshot["lineage_id"].value_counts().to_numpy(dtype=float)
    return counts / counts.sum() if counts.sum() else counts


def _valid_diversity(snapshot: pd.DataFrame, *, validity_threshold: float) -> float:
    valid = snapshot[snapshot["validity"] >= validity_threshold]
    cols = [c for c in ["validity", "novelty", "alignment"] if c in valid.columns]
    if len(valid) < 2 or not cols:
        return 0.0
    x = valid[cols].to_numpy(dtype=float)
    distances = []
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            distances.append(float(np.linalg.norm(x[i] - x[j]) / np.sqrt(len(cols))))
    return float(np.clip(np.mean(distances), 0.0, 1.0))


def compute_final_metrics_from_frames(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    events = frames.get("events", pd.DataFrame())
    ideas = frames.get("idea_states", pd.DataFrame())
    humans = frames.get("human_states", pd.DataFrame())
    ais = frames.get("ai_states", pd.DataFrame())
    platforms = frames.get("platform_states", pd.DataFrame())
    if events.empty or ideas.empty or humans.empty or ais.empty or platforms.empty:
        return pd.DataFrame()

    interactions = events[events["event_type"].eq("interaction_cycle")].copy()
    if "is_intervention" in interactions.columns:
        interactions = interactions[~interactions["is_intervention"].fillna(False)]
    if interactions.empty:
        return pd.DataFrame()

    cfg = MetricsPipelineConfig()
    final_time = int(interactions["time"].max())
    run_id = str(interactions.iloc[0]["run_id"])
    scenario_id = str(interactions.iloc[0]["scenario_id"])
    times = sorted(interactions["time"].unique())
    baseline_end = times[min(len(times), cfg.baseline_window) - 1]
    window_start = max(min(times), final_time - cfg.window_size + 1)
    window = interactions[(interactions["time"] >= window_start) & (interactions["time"] <= final_time)]

    initial_validity = float(interactions.iloc[0]["human_input_validity"])
    initial_alignment = float(interactions.iloc[0].get("human_input_alignment", 1.0))
    snapshot = _final_idea_snapshot(ideas, time=final_time)
    human = _state_latest(humans, time=final_time, phase="post")
    ai = _state_latest(ais, time=final_time, phase="post")
    platform = _state_latest(platforms, time=final_time)

    rep_rate = repetition_rate(
        window["repetition_score"].fillna(0).to_numpy(dtype=float),
        threshold=cfg.repetition_threshold,
    )
    mean_validity = _mean(window["final_validity"], initial_validity)
    mean_novelty = _mean(window["final_novelty"], 0.0)
    mean_alignment = _mean(window["alignment"], initial_alignment)
    mean_error = _mean(window["error_severity"], 0.0)
    mean_distortion = _mean(window.get("distortion_score", pd.Series(dtype=float)), 0.0)
    mean_ai_origin = _mean(window.get("ai_origin_share", pd.Series(dtype=float)), 0.0)
    mean_co_thinking = _mean(window.get("co_thinking", window.get("co_thinking_score", pd.Series(dtype=float))), 0.0)
    mean_qcm = _mean(window.get("co_movement", window.get("co_movement_score", pd.Series(dtype=float))), 0.0)

    lineage_p = _lineage_probabilities(snapshot)
    concentration = lineage_concentration(lineage_p)
    effective_diversity = effective_lineage_diversity(lineage_p)
    valid_diversity = _valid_diversity(snapshot, validity_threshold=cfg.validity_threshold)
    valid_survival = (
        float(np.mean(snapshot["validity"].to_numpy(dtype=float) >= cfg.validity_threshold))
        if not snapshot.empty
        else 0.0
    )
    baseline = ideas[ideas["time"] <= baseline_end].copy()
    frequency = baseline["descendant_count"].fillna(0).to_numpy(dtype=float) + 1.0 if "descendant_count" in baseline.columns else []
    alive = baseline["idea_id"].isin(set(snapshot["idea_id"])).to_numpy(dtype=bool) if not baseline.empty else []
    tail_survival = tail_idea_survival(frequency, alive, quantile=cfg.tail_quantile) if len(frequency) else 1.0

    reproduction = reproduction_numbers(
        snapshot[["idea_id", "parent_idea_id", "validity", "novelty", "error_severity"]],
        validity_threshold=cfg.validity_threshold,
        novelty_threshold=cfg.novelty_threshold,
        error_threshold=cfg.error_threshold,
    ) if not snapshot.empty else {"R_V": 0.0, "R_N": 0.0, "R_E": 0.0}
    r_v = reproduction["R_V"]
    r_n = reproduction["R_N"]
    r_e = reproduction["R_E"]
    ner = net_epistemic_reproduction(r_v, r_e)

    independent_performance = float(human.get("independent_performance", mean_validity)) if human is not None else mean_validity
    human_reliance = float(human.get("ai_reliance", mean_ai_origin)) if human is not None else mean_ai_origin
    verification = float(human.get("verification", 0.0)) if human is not None else 0.0
    context_contamination = float(ai.get("context_contamination", 0.0)) if ai is not None else 0.0
    model_contamination = float(platform.get("model_contamination", 0.0)) if platform is not None else 0.0
    platform_performance = float(platform.get("platform_performance", mean_validity)) if platform is not None else mean_validity

    valid_creativity = float(np.clip(mean_novelty * mean_validity * mean_alignment, 0.0, 1.0))
    utility_gain = float(np.clip(mean_validity - initial_validity + 0.5, 0.0, 1.0))
    preservation = preservation_score(
        valid_survival=valid_survival,
        intention_survival=mean_alignment,
        independent_human_performance=independent_performance,
    )
    expansion = productive_expansion_score(
        valid_diversity=valid_diversity,
        valid_creativity=valid_creativity,
        utility_gain=utility_gain,
    )
    development = co_development_score(preservation, expansion)
    baseline_novelty = _mean(interactions[interactions["time"] <= baseline_end]["final_novelty"], mean_novelty)
    diversity_loss = float(np.clip(baseline_novelty - mean_novelty, 0.0, 1.0))
    intention_drift = float(np.clip(initial_alignment - mean_alignment, 0.0, 1.0))
    performance_decline = float(np.clip(initial_validity - mean_validity, 0.0, 1.0))
    capability_decline = float(np.clip(initial_validity - independent_performance, 0.0, 1.0))
    degradation = co_degradation_score(
        error_growth=mean_error,
        diversity_loss=diversity_loss,
        dependence_growth=human_reliance,
        intention_drift=intention_drift,
        independent_capability_decline=capability_decline,
    )
    external_capacity = float(np.clip(_mean(window.get("source_independence", pd.Series(dtype=float)), 0.0), 0.0, 1.0))
    recovery = recovery_probability(
        valid_reproduction=r_v,
        error_reproduction=r_e,
        lineage_diversity=effective_diversity,
        verification_capacity=verification,
        external_input_capacity=external_capacity,
    )
    risk = recursive_degradation_risk(
        lineage_concentration=concentration,
        diversity_loss=diversity_loss,
        error_persistence_value=mean_error,
        error_reproduction=r_e,
        intention_drift=intention_drift,
        performance_decline=performance_decline,
        human_reliance=human_reliance,
        recovery_probability_value=recovery,
        tail_survival=tail_survival,
    )
    extinction_risk = co_extinction_risk(
        valid_vanishing=1.0 - valid_survival,
        distortion=mean_distortion,
        repetition=rep_rate,
        error_persistence_value=mean_error,
        performance_decline=performance_decline,
        recovery_probability_value=recovery,
    )
    confidence = window["ai_confidence"] if "ai_confidence" in window.columns else pd.Series(dtype=float)
    calibration = calibration_error(
        confidence.to_numpy(dtype=float),
        window["ai_output_validity"].to_numpy(dtype=float),
    ) if len(confidence) else 0.0

    return pd.DataFrame([{
        "run_id": run_id,
        "scenario_id": scenario_id,
        "time": final_time,
        "window_start": int(window_start),
        "window_size": int(len(window)),
        "co_thinking": mean_co_thinking,
        "quality_adjusted_comovement": mean_qcm,
        "mean_validity": mean_validity,
        "mean_novelty": mean_novelty,
        "mean_alignment": mean_alignment,
        "mean_error_severity": mean_error,
        "calibration_error": calibration,
        "repetition_rate": rep_rate,
        "distortion_rate": mean_distortion,
        "lineage_concentration": concentration,
        "effective_lineage_diversity": effective_diversity,
        "valid_diversity": valid_diversity,
        "valid_idea_survival": valid_survival,
        "tail_idea_survival": tail_survival,
        "R_V": r_v,
        "R_N": r_n,
        "R_E": r_e,
        "NER": ner,
        "preservation": preservation,
        "productive_expansion": expansion,
        "co_development_score": development,
        "co_degradation_score": degradation,
        "recovery_probability": recovery,
        "co_extinction_risk": extinction_risk,
        "recursive_degradation_risk": risk,
        "intervention_urgency": intervention_urgency(
            current_risk=risk,
            previous_risk=0.0,
            risk_two_steps_back=0.0,
        ),
        "human_ai_reliance": human_reliance,
        "human_independent_performance": independent_performance,
        "context_contamination": context_contamination,
        "model_contamination": model_contamination,
        "platform_performance": platform_performance,
        "source_independence": external_capacity,
    }])


def run_single_scenario(*, scenario_path: str | Path, steps: int, seed: int, run_id: str, controls: dict[str, Any] | None = None, final_metrics_only: bool = False) -> dict[str, pd.DataFrame]:
    scenario = load_scenario_with_overrides(scenario_path, controls)
    config = build_config(
        scenario_id=scenario.id,
        run_id=run_id,
        steps=steps,
        seed=seed,
        controls=controls,
    )
    result = run_recursive_interaction(
        config,
        scenario_runtime=ScenarioRuntime(
            scenario,
            max_interventions=controls.get("max_interventions"),
        ),
    )
    frames = result.as_frames()
    frames["outcome_metrics"] = (
        compute_final_metrics_from_frames(frames)
        if final_metrics_only
        else compute_metrics_from_frames(frames)
    )
    return frames


def read_stage_outputs(input_dir: str | Path) -> dict[str, pd.DataFrame]:
    root = Path(input_dir)
    frames = {}
    for name in OUTPUT_TABLES + ("outcome_metrics",):
        path = root / f"{name}.parquet"
        frames[name] = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    return frames


def save_table(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


def combine_frame_list(frames: list[pd.DataFrame]) -> pd.DataFrame:
    nonempty = [f for f in frames if f is not None and not f.empty]
    return pd.concat(nonempty, ignore_index=True) if nonempty else pd.DataFrame()
