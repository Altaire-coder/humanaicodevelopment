from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt

from .styles import save_figure


def plot_intervention_counts(
    intervention_counts: pd.DataFrame,
    output_dir,
    *,
    vector_format: str = "pdf",
) -> None:
    if intervention_counts.empty:
        return

    data = intervention_counts.copy()
    count_col = "count" if "count" in data.columns else data.columns[-1]

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    labels = data["scenario_id"].astype(str) + "\n" + data["event_type"].astype(str)
    ax.bar(labels, data[count_col])
    ax.set_title("Intervention counts")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=25)
    save_figure(fig, output_dir, "fig_06_intervention_counts", vector_format=vector_format)
    plt.close(fig)


def plot_intervention_timing(
    intervention_timing: pd.DataFrame,
    output_dir,
    *,
    vector_format: str = "pdf",
) -> None:
    if intervention_timing.empty:
        return

    data = intervention_timing.copy()
    if "first_time" not in data.columns:
        return

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for event_type, group in data.groupby("event_type", sort=True):
        ax.hist(group["first_time"], bins=20, alpha=0.55, label=event_type)

    ax.set_title("Distribution of first intervention timing")
    ax.set_xlabel("First intervention time")
    ax.set_ylabel("Number of runs")
    ax.legend(frameon=False)
    save_figure(fig, output_dir, "fig_07_intervention_timing", vector_format=vector_format)
    plt.close(fig)
