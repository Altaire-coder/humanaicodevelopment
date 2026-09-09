from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd


_RUN_SUFFIX_PATTERN = re.compile(r"(R\d+)$")


def _ci95(series: pd.Series) -> float:
    values = series.dropna().to_numpy(dtype=float)
    if len(values) <= 1:
        return 0.0
    return float(1.96 * np.std(values, ddof=1) / math.sqrt(len(values)))


def infer_matched_run_id(run_id: str) -> str:
    value = str(run_id)
    match = _RUN_SUFFIX_PATTERN.search(value)
    if match:
        return match.group(1)
    return value


def add_matched_run_id(
    frame: pd.DataFrame,
    *,
    run_col: str = "run_id",
    matched_col: str = "matched_run_id",
) -> pd.DataFrame:
    out = frame.copy()
    if matched_col not in out.columns:
        out[matched_col] = out[run_col].map(infer_matched_run_id)
    return out


def summarize_metric(
    frame: pd.DataFrame,
    *,
    group_cols: list[str],
    metric: str,
) -> pd.DataFrame:
    grouped = frame.groupby(group_cols, dropna=False)[metric]
    out = grouped.agg(["mean", "std", "count"]).reset_index()
    out["se"] = out["std"] / np.sqrt(out["count"].clip(lower=1))
    out["ci95"] = grouped.apply(_ci95).reset_index(drop=True)
    out["metric"] = metric
    return out[
        group_cols + ["metric", "mean", "std", "se", "ci95", "count"]
    ]


def final_state_summary(
    metrics: pd.DataFrame,
    *,
    metrics_to_summarize: list[str],
) -> pd.DataFrame:
    final = (
        metrics.sort_values("time")
        .groupby(["scenario_id", "run_id"], as_index=False)
        .tail(1)
    )
    pieces = [
        summarize_metric(
            final,
            group_cols=["scenario_id"],
            metric=metric,
        )
        for metric in metrics_to_summarize
        if metric in final.columns
    ]
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


def last_window_summary(
    metrics: pd.DataFrame,
    *,
    metrics_to_summarize: list[str],
    last_n: int = 20,
) -> pd.DataFrame:
    max_time = metrics.groupby(["scenario_id", "run_id"])["time"].transform("max")
    window = metrics[metrics["time"] >= max_time - last_n + 1].copy()

    run_means = (
        window.groupby(["scenario_id", "run_id"], as_index=False)[
            [m for m in metrics_to_summarize if m in window.columns]
        ]
        .mean()
    )

    pieces = [
        summarize_metric(
            run_means,
            group_cols=["scenario_id"],
            metric=metric,
        )
        for metric in metrics_to_summarize
        if metric in run_means.columns
    ]
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


def scenario_contrast(
    metrics: pd.DataFrame,
    *,
    baseline_scenario: str,
    metric: str,
    final_only: bool = True,
    matched_run_col: str = "matched_run_id",
) -> pd.DataFrame:
    if final_only:
        frame = (
            metrics.sort_values("time")
            .groupby(["scenario_id", "run_id"], as_index=False)
            .tail(1)
        )
    else:
        frame = metrics.copy()

    values = frame[["scenario_id", "run_id", metric]].dropna()
    values = add_matched_run_id(
        values,
        run_col="run_id",
        matched_col=matched_run_col,
    )

    base = values[values["scenario_id"] == baseline_scenario][
        [matched_run_col, metric]
    ].rename(columns={metric: "baseline"})

    joined = values.merge(base, on=matched_run_col, how="inner")
    joined["delta"] = joined[metric] - joined["baseline"]

    out = (
        joined.groupby("scenario_id")["delta"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    out["se"] = out["std"] / np.sqrt(out["count"].clip(lower=1))
    out["ci95"] = (
        joined.groupby("scenario_id")["delta"]
        .apply(_ci95)
        .reset_index(drop=True)
    )
    out["metric"] = metric
    out["baseline_scenario"] = baseline_scenario
    out["matched_runs"] = out["count"]
    return out[
        [
            "baseline_scenario",
            "scenario_id",
            "metric",
            "mean",
            "std",
            "se",
            "ci95",
            "count",
            "matched_runs",
        ]
    ]
