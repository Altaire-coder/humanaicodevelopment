from __future__ import annotations

import pandas as pd


def parameter_sensitivity_placeholder() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "parameter",
            "metric",
            "sensitivity_index",
            "method",
            "notes",
        ]
    )


def ablation_summary(
    results: pd.DataFrame,
    *,
    baseline_label: str,
    ablation_col: str = "ablation",
    metric: str = "co_development_score",
) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()

    baseline = results[results[ablation_col] == baseline_label][metric].mean()
    out = (
        results.groupby(ablation_col, as_index=False)[metric]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    out["delta_from_baseline"] = out["mean"] - baseline
    out["metric"] = metric
    return out
