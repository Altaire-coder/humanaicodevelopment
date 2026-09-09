from __future__ import annotations

import pandas as pd


def time_to_threshold(
    metrics: pd.DataFrame,
    *,
    metric: str,
    threshold: float,
    direction: str = "above",
) -> pd.DataFrame:
    if direction not in {"above", "below"}:
        raise ValueError("direction must be 'above' or 'below'.")

    rows = []
    for (scenario_id, run_id), group in metrics.groupby(["scenario_id", "run_id"]):
        group = group.sort_values("time")
        if direction == "above":
            hits = group[group[metric] >= threshold]
        else:
            hits = group[group[metric] <= threshold]

        if hits.empty:
            rows.append(
                {
                    "scenario_id": scenario_id,
                    "run_id": run_id,
                    "metric": metric,
                    "threshold": threshold,
                    "direction": direction,
                    "event_observed": False,
                    "time_to_event": int(group["time"].max()),
                }
            )
        else:
            rows.append(
                {
                    "scenario_id": scenario_id,
                    "run_id": run_id,
                    "metric": metric,
                    "threshold": threshold,
                    "direction": direction,
                    "event_observed": True,
                    "time_to_event": int(hits.iloc[0]["time"]),
                }
            )
    return pd.DataFrame(rows)


def survival_summary(tt: pd.DataFrame) -> pd.DataFrame:
    return (
        tt.groupby("scenario_id", as_index=False)
        .agg(
            event_rate=("event_observed", "mean"),
            mean_time_to_event=("time_to_event", "mean"),
            median_time_to_event=("time_to_event", "median"),
            n=("run_id", "count"),
        )
    )
