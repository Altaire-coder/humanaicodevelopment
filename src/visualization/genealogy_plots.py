from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt

from .styles import save_figure


def plot_genealogy_proxy(
    lineage_edges: pd.DataFrame,
    idea_states: pd.DataFrame,
    output_dir,
    *,
    scenario_id: str = "S0_no_treatment",
    run_id: str | None = None,
    vector_format: str = "pdf",
) -> None:
    if lineage_edges.empty or idea_states.empty:
        return

    ideas = idea_states[idea_states["scenario_id"] == scenario_id].copy()
    edges = lineage_edges[lineage_edges["scenario_id"] == scenario_id].copy()

    if run_id is None:
        if ideas.empty:
            return
        run_id = str(ideas["run_id"].iloc[0])

    ideas = ideas[ideas["run_id"] == run_id].copy()
    edges = edges[edges["run_id"] == run_id].copy()

    if ideas.empty:
        return

    # Lightweight genealogy proxy: plot idea validity and error over time.
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    if "validity" in ideas.columns:
        ax.scatter(ideas["time"], ideas["validity"], s=22, alpha=0.75, label="validity")
    if "error_severity" in ideas.columns:
        ax.scatter(ideas["time"], ideas["error_severity"], s=22, alpha=0.75, label="error severity")

    ax.set_title(f"Idea genealogy proxy: {scenario_id}, {run_id}")
    ax.set_xlabel("Generation/time")
    ax.set_ylabel("Idea attribute")
    ax.legend(frameon=False)
    safe_run = str(run_id).replace("/", "_")
    save_figure(fig, output_dir, f"fig_03_genealogy_proxy_{safe_run}", vector_format=vector_format)
    plt.close(fig)
