from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .labels import scenario_label
from .styles import save_figure


def plot_transition_matrix_heatmap(
    transition_summary: pd.DataFrame,
    output_dir: str | Path,
    *,
    vector_format: str = "pdf",
) -> None:
    if transition_summary.empty:
        return
    transitions = ["P_VV", "P_VE", "P_EV", "P_EE"]
    scenarios = list(transition_summary["scenario_id"].drop_duplicates())
    data = (
        transition_summary.pivot_table(
            index="scenario_id",
            columns="transition",
            values="mean",
            aggfunc="first",
        )
        .reindex(index=scenarios, columns=transitions)
        .fillna(0.0)
    )

    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    im = ax.imshow(data.to_numpy(dtype=float), vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_xticks(range(len(transitions)))
    ax.set_xticklabels([r"$P_{VV}$", r"$P_{VE}$", r"$P_{EV}$", r"$P_{EE}$"], fontsize=8)
    ax.set_yticks(range(len(scenarios)))
    ax.set_yticklabels([scenario_label(sid) for sid in scenarios], fontsize=8)
    for row in range(data.shape[0]):
        for col in range(data.shape[1]):
            value = data.iloc[row, col]
            color = "black" if value > 0.55 else "white"
            ax.text(col, row, f"{value:.2f}", ha="center", va="center", color=color, fontsize=8)
    ax.set_xlabel("Transition type")
    ax.set_title("Epistemic parent-child transitions by scenario", pad=8)
    ax.tick_params(length=0)
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.025, label="Mean transition probability")
    fig.tight_layout()
    save_figure(fig, Path(output_dir), "fig_15_transition_matrix_heatmap", vector_format=vector_format)
    plt.close(fig)


def plot_phase_diagram_with_thresholds(
    outcome_metrics: pd.DataFrame,
    output_dir: str | Path,
    *,
    risk_threshold: float = 0.20,
    development_threshold: float = 0.50,
    sample_per_scenario: int = 2500,
    vector_format: str = "pdf",
) -> None:
    if outcome_metrics.empty:
        return
    x = "recursive_degradation_risk"
    y = "co_development_score"
    scenarios = list(outcome_metrics["scenario_id"].drop_duplicates())
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.4), sharex=True, sharey=True)
    axes = axes.flatten()
    xmin, xmax = outcome_metrics[x].min(), outcome_metrics[x].max()
    ymin, ymax = outcome_metrics[y].min(), outcome_metrics[y].max()
    for ax, scenario_id in zip(axes, scenarios):
        group = outcome_metrics[outcome_metrics["scenario_id"].eq(scenario_id)].copy()
        if len(group) > sample_per_scenario:
            group = group.sample(sample_per_scenario, random_state=42)
        ax.axvspan(xmin, risk_threshold, ymin=0, ymax=1, color="#DDEFE5", alpha=0.28, zorder=0)
        ax.axhspan(development_threshold, ymax, xmin=0, xmax=1, color="#E8F1FB", alpha=0.28, zorder=0)
        ax.axvline(risk_threshold, color="#444444", linestyle="--", linewidth=0.8)
        ax.axhline(development_threshold, color="#444444", linestyle="--", linewidth=0.8)
        ax.scatter(group[x], group[y], s=6, alpha=0.20)
        ax.text(
            risk_threshold,
            ymax,
            "risk threshold",
            ha="right",
            va="top",
            fontsize=7,
            rotation=90,
            color="#444444",
        )
        ax.text(
            xmax,
            development_threshold,
            "co-development threshold",
            ha="right",
            va="bottom",
            fontsize=7,
            color="#444444",
        )
        ax.set_title(scenario_label(scenario_id))
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
    for ax in axes[len(scenarios):]:
        ax.axis("off")
    for ax in axes[::2]:
        ax.set_ylabel("Co-development")
    for ax in axes[-2:]:
        ax.set_xlabel("Recursive degradation risk")
    fig.suptitle("Co-development-recursive-degradation phase space by scenario")
    fig.tight_layout()
    save_figure(fig, Path(output_dir), "fig_14_phase_diagram_faceted", vector_format=vector_format)
    plt.close(fig)
