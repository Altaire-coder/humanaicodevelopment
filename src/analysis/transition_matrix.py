from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


TRANSITIONS = ["P_VV", "P_VE", "P_EV", "P_EE"]


def compute_transition_matrices(
    events: pd.DataFrame,
    *,
    validity_threshold: float = 0.60,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if events.empty:
        return pd.DataFrame(), pd.DataFrame()

    required = {
        "scenario_id",
        "run_id",
        "time",
        "event_type",
        "human_input_validity",
        "final_validity",
    }
    missing = required - set(events.columns)
    if missing:
        raise ValueError(f"events table is missing required columns: {sorted(missing)}")

    interactions = events[events["event_type"].eq("interaction_cycle")].copy()
    if "is_intervention" in interactions.columns:
        interactions = interactions[~interactions["is_intervention"].fillna(False)]
    if interactions.empty:
        return pd.DataFrame(), pd.DataFrame()

    interactions["parent_state"] = np.where(
        interactions["human_input_validity"].astype(float) >= validity_threshold,
        "V",
        "E",
    )
    interactions["child_state"] = np.where(
        interactions["final_validity"].astype(float) >= validity_threshold,
        "V",
        "E",
    )
    interactions["transition"] = (
        "P_"
        + interactions["parent_state"].astype(str)
        + interactions["child_state"].astype(str)
    )

    count_cols = ["scenario_id", "run_id", "transition"]
    counts = interactions.groupby(count_cols, as_index=False).size().rename(
        columns={"size": "count"}
    )
    wide_counts = (
        counts.pivot_table(
            index=["scenario_id", "run_id"],
            columns="transition",
            values="count",
            fill_value=0,
            aggfunc="sum",
        )
        .reset_index()
    )
    for transition in TRANSITIONS:
        if transition not in wide_counts.columns:
            wide_counts[transition] = 0

    valid_total = wide_counts["P_VV"] + wide_counts["P_VE"]
    error_total = wide_counts["P_EV"] + wide_counts["P_EE"]
    run_rows = wide_counts[["scenario_id", "run_id"]].copy()
    run_rows["n_parent_valid"] = valid_total
    run_rows["n_parent_error"] = error_total
    run_rows["P_VV"] = np.where(valid_total > 0, wide_counts["P_VV"] / valid_total, np.nan)
    run_rows["P_VE"] = np.where(valid_total > 0, wide_counts["P_VE"] / valid_total, np.nan)
    run_rows["P_EV"] = np.where(error_total > 0, wide_counts["P_EV"] / error_total, np.nan)
    run_rows["P_EE"] = np.where(error_total > 0, wide_counts["P_EE"] / error_total, np.nan)

    rows = []
    for scenario_id, group in run_rows.groupby("scenario_id"):
        base = {
            "scenario_id": scenario_id,
            "runs": int(group["run_id"].nunique()),
            "valid_parent_edges": int(group["n_parent_valid"].sum()),
            "error_parent_edges": int(group["n_parent_error"].sum()),
        }
        for transition in TRANSITIONS:
            values = group[transition].dropna()
            se = values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0.0
            rows.append(
                {
                    **base,
                    "transition": transition,
                    "mean": float(values.mean()) if len(values) else np.nan,
                    "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                    "ci95": float(1.96 * se),
                    "n_runs_with_denominator": int(len(values)),
                }
            )

    return run_rows, pd.DataFrame(rows)


def _write_latex(summary: pd.DataFrame, output: Path) -> None:
    if summary.empty:
        return
    pivot = summary.pivot_table(
        index="scenario_id",
        columns="transition",
        values="mean",
        aggfunc="first",
    ).reset_index()
    pivot.to_latex(output, index=False, float_format="%.4f")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", default="outputs/stage6/events.parquet")
    parser.add_argument("--output-dir", default="outputs/validation")
    parser.add_argument("--validity-threshold", type=float, default=0.60)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    events = pd.read_parquet(args.events)
    run_level, summary = compute_transition_matrices(
        events,
        validity_threshold=args.validity_threshold,
    )
    run_level.to_csv(output_dir / "transition_matrix_run_level.csv", index=False)
    summary.to_csv(output_dir / "transition_matrix_summary.csv", index=False)
    _write_latex(summary, output_dir / "transition_matrix_summary.tex")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
