from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

from .accuracy import calibration_error
from .core_metrics import quality_adjusted_comovement
from .development import (
    co_degradation_score,
    co_development_score,
    net_epistemic_reproduction,
    preservation_score,
    productive_expansion_score,
)
from .diversity import (
    effective_lineage_diversity,
    lineage_concentration,
    repetition_rate,
    tail_idea_survival,
)
from .extinction import co_extinction_risk, recovery_probability
from .genealogy import reproduction_numbers
from .risk import intervention_urgency, recursive_degradation_risk


@dataclass(frozen=True)
class MetricsPipelineConfig:
    window_size: int = 10
    validity_threshold: float = 0.60
    novelty_threshold: float = 0.70
    error_threshold: float = 0.40
    repetition_threshold: float = 0.80
    tail_quantile: float = 0.25
    baseline_window: int = 5
    show_progress: bool = False

    def validate(self) -> None:
        if self.window_size <= 0:
            raise ValueError("window_size must be positive.")
        if self.baseline_window <= 0:
            raise ValueError("baseline_window must be positive.")
        for name in [
            "validity_threshold",
            "novelty_threshold",
            "error_threshold",
            "repetition_threshold",
            "tail_quantile",
        ]:
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0,1].")


def _safe_mean(values: Iterable[float], default: float = 0.0) -> float:
    values = list(values)
    return float(np.mean(values)) if values else float(default)


def _interaction_events(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return events.copy()
    mask = events["event_type"].eq("interaction_cycle")
    if "is_intervention" in events.columns:
        mask &= ~events["is_intervention"].fillna(False)
    return events.loc[mask].copy()


def _state_at_or_before(
    frame: pd.DataFrame,
    *,
    run_id: str,
    scenario_id: str,
    time: int,
    phase: str | None = None,
) -> pd.Series | None:
    if frame.empty:
        return None
    subset = frame[
        (frame["run_id"] == run_id)
        & (frame["scenario_id"] == scenario_id)
        & (frame["time"] <= time)
    ]
    if phase is not None and "state_phase" in subset.columns:
        subset = subset[subset["state_phase"] == phase]
    if subset.empty:
        return None
    return subset.sort_values("time").iloc[-1]


def _active_idea_snapshot(
    ideas: pd.DataFrame,
    *,
    run_id: str,
    scenario_id: str,
    time: int,
) -> pd.DataFrame:
    subset = ideas[
        (ideas["run_id"] == run_id)
        & (ideas["scenario_id"] == scenario_id)
        & (ideas["time"] <= time)
    ].copy()
    if subset.empty:
        return subset
    subset = (
        subset.sort_values(["idea_id", "time"])
        .groupby("idea_id", as_index=False)
        .tail(1)
    )
    if "alive" in subset.columns:
        return subset[subset["alive"].fillna(True)].copy()
    return subset


def _lineage_probabilities(snapshot: pd.DataFrame) -> np.ndarray:
    if snapshot.empty or "lineage_id" not in snapshot.columns:
        return np.asarray([], dtype=float)
    counts = snapshot["lineage_id"].value_counts().to_numpy(dtype=float)
    return counts / counts.sum() if counts.sum() else counts


def _valid_diversity_proxy(snapshot: pd.DataFrame, validity_threshold: float) -> float:
    valid = snapshot[snapshot["validity"] >= validity_threshold]
    cols = [c for c in ["validity", "novelty", "alignment"] if c in valid.columns]
    if len(valid) < 2 or not cols:
        return 0.0
    x = valid[cols].to_numpy(dtype=float)
    pairwise = []
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            pairwise.append(float(np.linalg.norm(x[i] - x[j]) / np.sqrt(len(cols))))
    return float(np.clip(np.mean(pairwise), 0.0, 1.0))


def _tail_survival(
    all_ideas: pd.DataFrame,
    snapshot: pd.DataFrame,
    *,
    baseline_end: int,
    quantile: float,
) -> float:
    baseline = all_ideas[all_ideas["time"] <= baseline_end].copy()
    if baseline.empty:
        return 1.0

    if "descendant_count" in baseline.columns:
        frequency = baseline["descendant_count"].fillna(0).to_numpy(dtype=float) + 1.0
    else:
        frequency = (
            baseline["lineage_id"].map(baseline["lineage_id"].value_counts())
            .to_numpy(dtype=float)
        )

    current_ids = set(snapshot["idea_id"]) if not snapshot.empty else set()
    alive_mask = baseline["idea_id"].isin(current_ids).to_numpy(dtype=bool)
    return tail_idea_survival(frequency, alive_mask, quantile=quantile)


def compute_run_metrics(
    *,
    events: pd.DataFrame,
    idea_states: pd.DataFrame,
    human_states: pd.DataFrame,
    ai_states: pd.DataFrame,
    platform_states: pd.DataFrame,
    config: MetricsPipelineConfig | None = None,
) -> pd.DataFrame:
    cfg = config or MetricsPipelineConfig()
    cfg.validate()

    interactions = _interaction_events(events)
    if interactions.empty:
        return pd.DataFrame()

    rows: list[dict] = []

    groups = interactions[["run_id", "scenario_id"]].drop_duplicates()
    group_records = list(groups.itertuples(index=False))

    if cfg.show_progress and tqdm is not None:
        group_iter = tqdm(
            group_records,
            total=len(group_records),
            desc="compute metrics",
            unit="run",
        )
    else:
        group_iter = group_records

    for group in group_iter:
        run_id = group.run_id
        scenario_id = group.scenario_id

        if cfg.show_progress and tqdm is not None:
            group_iter.set_postfix(
                scenario=scenario_id,
                run=run_id,
                refresh=False,
            )

        e_run = interactions[
            (interactions["run_id"] == run_id)
            & (interactions["scenario_id"] == scenario_id)
        ].sort_values("time")
        i_run = idea_states[
            (idea_states["run_id"] == run_id)
            & (idea_states["scenario_id"] == scenario_id)
        ].sort_values("time")
        h_run = human_states[
            (human_states["run_id"] == run_id)
            & (human_states["scenario_id"] == scenario_id)
        ]
        a_run = ai_states[
            (ai_states["run_id"] == run_id)
            & (ai_states["scenario_id"] == scenario_id)
        ]
        p_run = platform_states[
            (platform_states["run_id"] == run_id)
            & (platform_states["scenario_id"] == scenario_id)
        ]

        times = sorted(e_run["time"].unique())
        initial_validity = float(e_run.iloc[0]["human_input_validity"])
        initial_alignment = float(e_run.iloc[0].get("human_input_alignment", 1.0))
        baseline_end = times[min(len(times), cfg.baseline_window) - 1]

        previous_risk = 0.0
        risk_two_back = 0.0

        for time in times:
            window_start = max(min(times), time - cfg.window_size + 1)
            ew = e_run[
                (e_run["time"] >= window_start)
                & (e_run["time"] <= time)
            ]
            snapshot = _active_idea_snapshot(
                i_run,
                run_id=run_id,
                scenario_id=scenario_id,
                time=time,
            )
            human = _state_at_or_before(
                h_run,
                run_id=run_id,
                scenario_id=scenario_id,
                time=time,
                phase="post",
            )
            ai = _state_at_or_before(
                a_run,
                run_id=run_id,
                scenario_id=scenario_id,
                time=time,
                phase="post",
            )
            platform = _state_at_or_before(
                p_run,
                run_id=run_id,
                scenario_id=scenario_id,
                time=time,
            )

            rep_rate = repetition_rate(
                ew["repetition_score"].fillna(0).to_numpy(dtype=float),
                threshold=cfg.repetition_threshold,
            )
            mean_validity = _safe_mean(ew["final_validity"], initial_validity)
            mean_novelty = _safe_mean(ew["final_novelty"], 0.0)
            mean_alignment = _safe_mean(ew["alignment"], initial_alignment)
            mean_error = _safe_mean(ew["error_severity"], 0.0)
            mean_distortion = _safe_mean(ew.get("distortion_score", pd.Series(dtype=float)), 0.0)
            mean_ai_origin = _safe_mean(ew.get("ai_origin_share", pd.Series(dtype=float)), 0.0)
            mean_co_thinking = _safe_mean(ew.get("co_thinking", ew.get("co_thinking_score", pd.Series(dtype=float))), 0.0)
            mean_qcm = _safe_mean(ew.get("co_movement", ew.get("co_movement_score", pd.Series(dtype=float))), 0.0)

            lineage_p = _lineage_probabilities(snapshot)
            concentration = lineage_concentration(lineage_p)
            effective_diversity = effective_lineage_diversity(lineage_p)

            valid_diversity = _valid_diversity_proxy(
                snapshot,
                cfg.validity_threshold,
            )
            valid_survival = (
                float(
                    np.mean(
                        snapshot["validity"].to_numpy(dtype=float)
                        >= cfg.validity_threshold
                    )
                )
                if not snapshot.empty
                else 0.0
            )
            intention_survival = mean_alignment
            tail_survival = _tail_survival(
                i_run,
                snapshot,
                baseline_end=baseline_end,
                quantile=cfg.tail_quantile,
            )

            reproduction = reproduction_numbers(
                snapshot[[
                    "idea_id",
                    "parent_idea_id",
                    "validity",
                    "novelty",
                    "error_severity",
                ]],
                validity_threshold=cfg.validity_threshold,
                novelty_threshold=cfg.novelty_threshold,
                error_threshold=cfg.error_threshold,
            ) if not snapshot.empty else {"R_V": 0.0, "R_N": 0.0, "R_E": 0.0}

            r_v = reproduction["R_V"]
            r_n = reproduction["R_N"]
            r_e = reproduction["R_E"]
            ner = net_epistemic_reproduction(r_v, r_e)

            independent_performance = (
                float(human.get("independent_performance", mean_validity))
                if human is not None else mean_validity
            )
            human_reliance = (
                float(human.get("ai_reliance", mean_ai_origin))
                if human is not None else mean_ai_origin
            )
            verification = (
                float(human.get("verification", 0.0))
                if human is not None else 0.0
            )
            context_contamination = (
                float(ai.get("context_contamination", 0.0))
                if ai is not None else 0.0
            )
            model_contamination = (
                float(platform.get("model_contamination", 0.0))
                if platform is not None else 0.0
            )
            platform_performance = (
                float(platform.get("platform_performance", mean_validity))
                if platform is not None else mean_validity
            )

            valid_creativity = float(np.clip(
                mean_novelty * mean_validity * mean_alignment,
                0.0,
                1.0,
            ))
            utility_gain = float(np.clip(
                mean_validity - initial_validity + 0.5,
                0.0,
                1.0,
            ))

            preservation = preservation_score(
                valid_survival=valid_survival,
                intention_survival=intention_survival,
                independent_human_performance=independent_performance,
            )
            expansion = productive_expansion_score(
                valid_diversity=valid_diversity,
                valid_creativity=valid_creativity,
                utility_gain=utility_gain,
            )
            development = co_development_score(preservation, expansion)

            baseline_novelty = _safe_mean(
                e_run[e_run["time"] <= baseline_end]["final_novelty"],
                mean_novelty,
            )
            diversity_loss = float(np.clip(
                baseline_novelty - mean_novelty,
                0.0,
                1.0,
            ))
            intention_drift = float(np.clip(
                initial_alignment - mean_alignment,
                0.0,
                1.0,
            ))
            performance_decline = float(np.clip(
                initial_validity - mean_validity,
                0.0,
                1.0,
            ))
            capability_decline = float(np.clip(
                initial_validity - independent_performance,
                0.0,
                1.0,
            ))

            degradation = co_degradation_score(
                error_growth=mean_error,
                diversity_loss=diversity_loss,
                dependence_growth=human_reliance,
                intention_drift=intention_drift,
                independent_capability_decline=capability_decline,
            )

            external_capacity = float(np.clip(
                _safe_mean(
                    ew.get("source_independence", pd.Series(dtype=float)),
                    0.0,
                ),
                0.0,
                1.0,
            ))
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
            urgency = intervention_urgency(
                current_risk=risk,
                previous_risk=previous_risk,
                risk_two_steps_back=risk_two_back,
            )
            extinction_risk = co_extinction_risk(
                valid_vanishing=1.0 - valid_survival,
                distortion=mean_distortion,
                repetition=rep_rate,
                error_persistence_value=mean_error,
                performance_decline=performance_decline,
                recovery_probability_value=recovery,
            )

            confidence_col = (
                ew["ai_confidence"]
                if "ai_confidence" in ew.columns
                else pd.Series(dtype=float)
            )
            calibration = calibration_error(
                confidence_col.to_numpy(dtype=float),
                ew["ai_output_validity"].to_numpy(dtype=float),
            ) if len(confidence_col) else 0.0

            rows.append({
                "run_id": run_id,
                "scenario_id": scenario_id,
                "time": int(time),
                "window_start": int(window_start),
                "window_size": int(len(ew)),
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
                "intervention_urgency": urgency,
                "human_ai_reliance": human_reliance,
                "human_independent_performance": independent_performance,
                "context_contamination": context_contamination,
                "model_contamination": model_contamination,
                "platform_performance": platform_performance,
                "source_independence": external_capacity,
            })

            risk_two_back = previous_risk
            previous_risk = risk

    return pd.DataFrame(rows)


def compute_metrics_from_directory(
    input_dir: str,
    output_path: str,
    config: MetricsPipelineConfig | None = None,
) -> pd.DataFrame:
    if config is None:
        config = MetricsPipelineConfig(show_progress=True)

    input_dir = str(input_dir)
    frames = {
        name: pd.read_parquet(f"{input_dir}/{name}.parquet")
        for name in [
            "events",
            "idea_states",
            "human_states",
            "ai_states",
            "platform_states",
        ]
    }

    result = compute_run_metrics(
        events=frames["events"],
        idea_states=frames["idea_states"],
        human_states=frames["human_states"],
        ai_states=frames["ai_states"],
        platform_states=frames["platform_states"],
        config=config,
    )
    result.to_parquet(output_path, index=False)
    return result
