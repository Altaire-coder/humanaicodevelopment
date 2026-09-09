from __future__ import annotations

import pandas as pd


def trajectory_summary(
    metrics: pd.DataFrame,
    *,
    metric: str,
) -> pd.DataFrame:
    if metric not in metrics.columns:
        raise KeyError(f"Metric not found: {metric}")

    grouped = metrics.groupby(["scenario_id", "time"], as_index=False)[metric]
    out = grouped.agg(["mean", "std", "count"]).reset_index()
    out["se"] = out["std"] / (out["count"] ** 0.5)
    out["ci95"] = 1.96 * out["se"]
    out["metric"] = metric
    return out[
        ["scenario_id", "time", "metric", "mean", "std", "se", "ci95", "count"]
    ]


def area_under_trajectory(
    metrics: pd.DataFrame,
    *,
    metric: str,
) -> pd.DataFrame:
    if metric not in metrics.columns:
        raise KeyError(f"Metric not found: {metric}")

    pieces = []
    for (scenario_id, run_id), group in metrics.groupby(["scenario_id", "run_id"]):
        group = group.sort_values("time")
        value = float(group[metric].sum())
        pieces.append(
            {
                "scenario_id": scenario_id,
                "run_id": run_id,
                "metric": metric,
                "auc_sum": value,
                "mean_over_time": float(group[metric].mean()),
            }
        )
    return pd.DataFrame(pieces)
