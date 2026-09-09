from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt

from .styles import save_figure


def plot_phase_diagram(
    outcome_metrics: pd.DataFrame,
    output_dir,
    *,
    x: str = "recursive_degradation_risk",
    y: str = "co_development_score",
    vector_format: str = "pdf",
) -> None:
    if outcome_metrics.empty or x not in outcome_metrics.columns or y not in outcome_metrics.columns:
        return

    sample = outcome_metrics.copy()
    if len(sample) > 10000:
        sample = sample.sample(10000, random_state=42)

    fig, ax = plt.subplots(figsize=(6.4, 5.2))

    for scenario_id, group in sample.groupby("scenario_id", sort=True):
        ax.scatter(group[x], group[y], s=8, alpha=0.35, label=scenario_id)

    ax.set_title("Convergence-divergence phase diagram")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.legend(frameon=False, markerscale=2)
    save_figure(fig, output_dir, "fig_04_phase_diagram", vector_format=vector_format)
    plt.close(fig)


def plot_phase_share(
    phase_share: pd.DataFrame,
    output_dir,
    *,
    vector_format: str = "pdf",
) -> None:
    if phase_share.empty:
        return

    pivot = phase_share.pivot_table(
        index="scenario_id",
        columns="phase",
        values="share",
        fill_value=0.0,
    )

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    bottom = None
    for phase in pivot.columns:
        values = pivot[phase]
        ax.bar(pivot.index, values, bottom=bottom, label=phase)
        bottom = values if bottom is None else bottom + values

    ax.set_title("Phase occupancy by scenario")
    ax.set_xlabel("Scenario")
    ax.set_ylabel("Share")
    ax.tick_params(axis="x", rotation=25)
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    save_figure(fig, output_dir, "fig_04b_phase_share", vector_format=vector_format)
    plt.close(fig)
