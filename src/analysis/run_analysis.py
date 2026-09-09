from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .change_points import simple_change_points
from .export_tables import export_table_bundle
from .intervention_analysis import (
    intervention_counts,
    intervention_timing,
    platform_update_summary,
)
from .io import ensure_outcome_metrics, read_stage_outputs
from .phase_analysis import assign_phase, phase_share
from .summarize_runs import (
    final_state_summary,
    last_window_summary,
    scenario_contrast,
)
from .survival_analysis import survival_summary, time_to_threshold
from .trajectory_models import area_under_trajectory, trajectory_summary


DEFAULT_METRICS = [
    "co_development_score",
    "co_degradation_score",
    "co_extinction_risk",
    "recursive_degradation_risk",
    "R_V",
    "R_N",
    "R_E",
    "NER",
    "recovery_probability",
    "mean_validity",
    "mean_novelty",
    "mean_alignment",
    "human_ai_reliance",
    "context_contamination",
    "model_contamination",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="outputs/stage6")
    parser.add_argument("--output-dir", default="outputs/analysis")
    parser.add_argument("--baseline-scenario", default="S0_no_treatment")
    parser.add_argument("--last-n", type=int, default=20)
    parser.add_argument("--risk-threshold", type=float, default=0.50)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = read_stage_outputs(input_dir)
    metrics = ensure_outcome_metrics(
        frames=frames,
        input_dir=input_dir,
        output_path=input_dir / "outcome_metrics.parquet",
    )

    metrics_to_use = [m for m in DEFAULT_METRICS if m in metrics.columns]

    tables: dict[str, pd.DataFrame] = {}

    tables["final_state_summary"] = final_state_summary(
        metrics,
        metrics_to_summarize=metrics_to_use,
    )
    tables["last_window_summary"] = last_window_summary(
        metrics,
        metrics_to_summarize=metrics_to_use,
        last_n=args.last_n,
    )

    contrast_pieces = [
        scenario_contrast(
            metrics,
            baseline_scenario=args.baseline_scenario,
            metric=metric,
            final_only=True,
        )
        for metric in metrics_to_use
    ]
    tables["scenario_contrasts"] = (
        pd.concat(contrast_pieces, ignore_index=True)
        if contrast_pieces
        else pd.DataFrame()
    )

    trajectory_pieces = [
        trajectory_summary(metrics, metric=metric)
        for metric in metrics_to_use
    ]
    tables["trajectory_summary"] = pd.concat(
        trajectory_pieces,
        ignore_index=True,
    )

    auc_pieces = [
        area_under_trajectory(metrics, metric=metric)
        for metric in metrics_to_use
    ]
    tables["trajectory_auc"] = pd.concat(auc_pieces, ignore_index=True)

    tables["change_points_recursive_risk"] = simple_change_points(
        metrics,
        metric="recursive_degradation_risk",
        min_time=5,
        z_threshold=2.0,
    )

    tt = time_to_threshold(
        metrics,
        metric="co_extinction_risk",
        threshold=args.risk_threshold,
        direction="above",
    )
    tables["time_to_co_extinction"] = tt
    tables["survival_summary"] = survival_summary(tt)

    phased = assign_phase(
        metrics,
        risk_threshold=args.risk_threshold,
        development_threshold=0.50,
    )
    tables["phase_share"] = phase_share(phased)

    tables["intervention_counts"] = intervention_counts(frames["events"])
    tables["intervention_timing"] = intervention_timing(frames["events"])
    tables["platform_update_summary"] = platform_update_summary(
        frames["platform_update_events"]
    )

    export_table_bundle(tables, output_dir)

    print(f"Saved analysis tables to {output_dir.resolve()}")
    for name, table in tables.items():
        print(f"{name}: {len(table):,} rows")


if __name__ == "__main__":
    main()
