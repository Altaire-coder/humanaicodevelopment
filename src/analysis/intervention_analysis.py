from __future__ import annotations

import pandas as pd


def intervention_counts(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty or "is_intervention" not in events.columns:
        return pd.DataFrame()

    inter = events[events["is_intervention"].fillna(False)].copy()
    if inter.empty:
        return pd.DataFrame(
            columns=["scenario_id", "event_type", "count"]
        )

    return (
        inter.groupby(["scenario_id", "event_type"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )


def intervention_timing(events: pd.DataFrame) -> pd.DataFrame:
    inter = events[events.get("is_intervention", False).fillna(False)].copy()
    if inter.empty:
        return pd.DataFrame(
            columns=[
                "scenario_id",
                "run_id",
                "event_type",
                "first_time",
                "last_time",
                "count",
            ]
        )

    return (
        inter.groupby(["scenario_id", "run_id", "event_type"], as_index=False)
        .agg(
            first_time=("time", "min"),
            last_time=("time", "max"),
            count=("time", "count"),
        )
    )


def platform_update_summary(platform_update_events: pd.DataFrame) -> pd.DataFrame:
    if platform_update_events.empty:
        return pd.DataFrame(
            columns=["scenario_id", "update_reason", "count", "mean_sample_size"]
        )

    return (
        platform_update_events.groupby(
            ["scenario_id", "update_reason"],
            as_index=False,
        )
        .agg(
            count=("time", "count"),
            mean_sample_size=("sample_size", "mean"),
        )
    )
