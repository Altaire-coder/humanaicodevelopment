from __future__ import annotations

import pandas as pd


def assign_phase(
    metrics: pd.DataFrame,
    *,
    risk_col: str = "recursive_degradation_risk",
    development_col: str = "co_development_score",
    risk_threshold: float = 0.50,
    development_threshold: float = 0.50,
) -> pd.DataFrame:
    out = metrics.copy()

    def label(row):
        high_risk = row[risk_col] >= risk_threshold
        high_dev = row[development_col] >= development_threshold
        if high_dev and not high_risk:
            return "productive_co_development"
        if high_dev and high_risk:
            return "productive_but_fragile"
        if (not high_dev) and high_risk:
            return "recursive_degradation"
        return "low_activity_stable"

    out["phase"] = out.apply(label, axis=1)
    return out


def phase_share(phased: pd.DataFrame) -> pd.DataFrame:
    counts = (
        phased.groupby(["scenario_id", "phase"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    total = counts.groupby("scenario_id")["count"].transform("sum")
    counts["share"] = counts["count"] / total
    return counts
