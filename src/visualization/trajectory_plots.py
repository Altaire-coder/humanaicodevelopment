from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt

from .styles import save_figure


def plot_untreated_trajectory(
    trajectory_summary: pd.DataFrame,
    output_dir,
    *,
    metric: str = "co_development_score",
    baseline_scenario: str = "S0_no_treatment",
    vector_format: str = "pdf",
) -> None:
    if trajectory_summary.empty:
        return

    data = trajectory_summary[
        (trajectory_summary["scenario_id"] == baseline_scenario)
        & (trajectory_summary["metric"] == metric)
    ].copy()

    if data.empty:
        return

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.plot(data["time"], data["mean"], label=baseline_scenario)
    if {"ci95"}.issubset(data.columns):
        ax.fill_between(
            data["time"],
            data["mean"] - data["ci95"],
            data["mean"] + data["ci95"],
            alpha=0.20,
        )

    ax.set_title(f"Untreated trajectory: {metric}")
    ax.set_xlabel("Time")
    ax.set_ylabel(metric)
    ax.legend(frameon=False)
    save_figure(fig, output_dir, f"fig_01_untreated_{metric}", vector_format=vector_format)
    plt.close(fig)


def plot_scenario_trajectories(
    trajectory_summary: pd.DataFrame,
    output_dir,
    *,
    metric: str = "co_development_score",
    vector_format: str = "pdf",
) -> None:
    if trajectory_summary.empty:
        return

    data = trajectory_summary[trajectory_summary["metric"] == metric].copy()
    if data.empty:
        return

    fig, ax = plt.subplots(figsize=(7.2, 4.2))

    for scenario_id, group in data.groupby("scenario_id", sort=True):
        group = group.sort_values("time")
        ax.plot(group["time"], group["mean"], label=scenario_id)
        if "ci95" in group.columns:
            ax.fill_between(
                group["time"],
                group["mean"] - group["ci95"],
                group["mean"] + group["ci95"],
                alpha=0.12,
            )

    ax.set_title(f"Scenario trajectory comparison: {metric}")
    ax.set_xlabel("Time")
    ax.set_ylabel(metric)
    ax.legend(frameon=False)
    save_figure(fig, output_dir, f"fig_05_scenario_trajectory_{metric}", vector_format=vector_format)
    plt.close(fig)


def plot_reproduction_numbers(
    trajectory_summary: pd.DataFrame,
    output_dir,
    *,
    vector_format: str = "pdf",
) -> None:
    if trajectory_summary.empty:
        return

    metrics = ["R_V", "R_E", "NER"]
    data = trajectory_summary[trajectory_summary["metric"].isin(metrics)].copy()
    if data.empty:
        return

    for scenario_id, group_s in data.groupby("scenario_id", sort=True):
        fig, ax = plt.subplots(figsize=(6.8, 4.0))
        for metric, group_m in group_s.groupby("metric", sort=False):
            group_m = group_m.sort_values("time")
            ax.plot(group_m["time"], group_m["mean"], label=metric)

        ax.set_title(f"Valid/error-bearing activity: {scenario_id}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Reproduction metric")
        ax.legend(frameon=False)
        safe_scenario = str(scenario_id).replace("/", "_")
        save_figure(
            fig,
            output_dir,
            f"fig_02_reproduction_numbers_{safe_scenario}",
            vector_format=vector_format,
        )
        plt.close(fig)
