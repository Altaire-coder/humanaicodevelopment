from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .common import CORE_METRICS, save_table


def _ci95(x: pd.Series) -> float:
    values = x.dropna().to_numpy(dtype=float)
    if len(values) <= 1:
        return 0.0
    return float(1.96 * np.std(values, ddof=1) / np.sqrt(len(values)))


def summarize_last_window(
    metrics: pd.DataFrame,
    *,
    last_n: int,
    metric_cols: list[str] | None = None,
) -> pd.DataFrame:
    metric_cols = metric_cols or CORE_METRICS
    available = [m for m in metric_cols if m in metrics.columns]
    if metrics.empty or not available:
        return pd.DataFrame()

    max_time = metrics.groupby(["scenario_id", "run_id"])["time"].transform("max")
    window = metrics[metrics["time"] >= max_time - last_n + 1].copy()

    run_means = (
        window.groupby(["scenario_id", "run_id"], as_index=False)[available]
        .mean()
    )

    rows = []
    for metric in available:
        for scenario_id, group in run_means.groupby("scenario_id"):
            rows.append(
                {
                    "last_n": last_n,
                    "scenario_id": scenario_id,
                    "metric": metric,
                    "mean": group[metric].mean(),
                    "std": group[metric].std(),
                    "ci95": _ci95(group[metric]),
                    "count": group[metric].count(),
                }
            )

    return pd.DataFrame(rows)


def robustness_across_windows(
    *,
    outcome_path: str | Path,
    output: str | Path,
    windows: list[int],
) -> pd.DataFrame:
    metrics = pd.read_parquet(outcome_path)
    pieces = [
        summarize_last_window(metrics, last_n=window)
        for window in windows
    ]
    out = pd.concat([p for p in pieces if not p.empty], ignore_index=True)
    save_table(out, output)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="outputs/stage6/outcome_metrics.parquet")
    parser.add_argument("--output", default="outputs/validation/robustness_window_summary.csv")
    parser.add_argument("--windows", nargs="+", type=int, default=[10, 20, 30])
    args = parser.parse_args()

    out = robustness_across_windows(
        outcome_path=args.input,
        output=args.output,
        windows=args.windows,
    )
    print(out.head(40).to_string(index=False))
    print(f"Saved window robustness summary to {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
