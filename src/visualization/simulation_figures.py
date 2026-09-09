from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import pandas as pd

from .conceptual_diagrams import (
    plot_conceptual_scheme,
    plot_intervention_mechanism_scheme,
)
from .contrast_plots import plot_final_metric_bar, plot_scenario_contrast_forest
from .descriptive_tables import export_descriptive_tables
from .io_helpers import load_analysis_outputs, load_stage_outputs, read_csv_if_exists
from .refined_intervention_plots import (
    plot_first_intervention_timing_strip,
    plot_intervention_frequency_per_run,
)
from .refined_trajectory_plots import (
    plot_error_reproduction_comparison,
    plot_scenario_trajectories_refined,
)
from .styles import apply_paper_style, ensure_dir
from .transition_plots import (
    plot_phase_diagram_with_thresholds,
    plot_transition_matrix_heatmap,
)


def build_simulation_outputs(
    input_dir: str | Path,
    analysis_dir: str | Path,
    output_dir: str | Path,
    burn_in: int = 5,
    vector_format: str = "pdf",
) -> None:
    apply_paper_style()
    output_root = ensure_dir(output_dir)
    figure_dir = ensure_dir(output_root / "figures")
    table_dir = ensure_dir(output_root / "tables")

    stage = load_stage_outputs(input_dir)
    analysis = load_analysis_outputs(analysis_dir)

    trajectory = analysis.get("trajectory_summary", pd.DataFrame())
    contrasts = analysis.get("scenario_contrasts", pd.DataFrame())
    final_summary = analysis.get("final_state_summary", pd.DataFrame())
    intervention_timing = analysis.get("intervention_timing", pd.DataFrame())
    outcome_metrics = stage.get("outcome_metrics", pd.DataFrame())
    events = stage.get("events", pd.DataFrame())
    transition_summary = read_csv_if_exists(Path(analysis_dir) / "transition_matrix_summary.csv")

    plot_conceptual_scheme(figure_dir, vector_format)
    plot_intervention_mechanism_scheme(figure_dir, vector_format)
    plot_scenario_trajectories_refined(
        trajectory,
        figure_dir,
        "co_development_score",
        burn_in,
        vector_format,
    )
    plot_scenario_trajectories_refined(
        trajectory,
        figure_dir,
        "recursive_degradation_risk",
        burn_in,
        vector_format,
    )
    plot_error_reproduction_comparison(trajectory, figure_dir, burn_in, vector_format)
    plot_scenario_contrast_forest(contrasts, figure_dir, vector_format=vector_format)
    plot_final_metric_bar(
        final_summary,
        figure_dir,
        "co_development_score",
        vector_format,
    )
    plot_final_metric_bar(
        final_summary,
        figure_dir,
        "recursive_degradation_risk",
        vector_format,
    )
    plot_transition_matrix_heatmap(
        transition_summary,
        figure_dir,
        vector_format=vector_format,
    )
    plot_intervention_frequency_per_run(events, figure_dir, vector_format)
    plot_first_intervention_timing_strip(
        intervention_timing,
        figure_dir,
        vector_format,
    )
    plot_phase_diagram_with_thresholds(
        outcome_metrics,
        figure_dir,
        vector_format=vector_format,
    )
    export_descriptive_tables(stage, analysis, table_dir)

    print(f"Saved simulation figures to {figure_dir.resolve()}")
    print(f"Saved simulation tables to {table_dir.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="outputs/results/simulation")
    parser.add_argument("--analysis-dir", default="outputs/results/analysis")
    parser.add_argument("--output-dir", default="outputs/results")
    parser.add_argument("--burn-in", type=int, default=5)
    parser.add_argument("--vector-format", choices=["pdf", "svg"], default="pdf")
    args = parser.parse_args()
    build_simulation_outputs(
        args.input_dir,
        args.analysis_dir,
        args.output_dir,
        args.burn_in,
        args.vector_format,
    )


if __name__ == "__main__":
    main()
