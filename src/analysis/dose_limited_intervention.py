from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from src.validation.common import CORE_METRICS, combine_frame_list, run_single_scenario


DOSE_ARMS: dict[str, int | None] = {
    "S2_first_1": 1,
    "S2_first_5": 5,
    "S2_first_10": 10,
    "S2_full": None,
}


def _summarize(metrics: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    final = metrics.sort_values("time").groupby(["dose_arm", "run_id"], as_index=False).tail(1)
    counts = (
        events[
            events.get("is_intervention", pd.Series(False, index=events.index)).fillna(False)
            & events.get("event_type", pd.Series("", index=events.index)).eq("critic_feedback")
        ]
        .groupby(["dose_arm", "run_id"])
        .size()
        .rename("intervention_count")
        .reset_index()
    )
    final = final.merge(counts, on=["dose_arm", "run_id"], how="left")
    final["intervention_count"] = final["intervention_count"].fillna(0)
    rows: list[dict[str, Any]] = []
    metrics_available = [metric for metric in CORE_METRICS if metric in final.columns]
    for arm, group in final.groupby("dose_arm"):
        row: dict[str, Any] = {
            "dose_arm": arm,
            "runs": int(group["run_id"].nunique()),
            "mean_interventions": float(group["intervention_count"].mean()),
        }
        for metric in metrics_available:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std())
        rows.append(row)
    order = {arm: idx for idx, arm in enumerate(DOSE_ARMS)}
    return pd.DataFrame(rows).sort_values("dose_arm", key=lambda s: s.map(order))


def _paper_table(summary: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "dose_arm",
        "mean_interventions",
        "co_development_score_mean",
        "recursive_degradation_risk_mean",
        "R_E_mean",
        "NER_mean",
    ]
    table = summary[columns].copy()
    table["mean_interventions"] = table["mean_interventions"].map(lambda x: f"{x:.2f}")
    for column in columns[2:]:
        table[column] = table[column].map(lambda x: f"{x:.4f}")
    return table.rename(
        columns={
            "dose_arm": "Dose arm",
            "mean_interventions": "Mean interventions",
            "co_development_score_mean": "Co-development",
            "recursive_degradation_risk_mean": "Recursive risk",
            "R_E_mean": "Error-bearing parent activity",
            "NER_mean": "NER",
        }
    )


def run_dose_limited_s2(
    *,
    output_dir: str | Path,
    scenario_path: str = "config/scenarios/S2_critic_feedback.yaml",
    steps: int = 100,
    runs: int = 100,
    base_seed: int = 20260700,
) -> pd.DataFrame:
    metrics_frames: list[pd.DataFrame] = []
    event_frames: list[pd.DataFrame] = []
    for arm, max_interventions in DOSE_ARMS.items():
        for run_index in range(runs):
            seed = base_seed + run_index
            run_id = f"{arm}_R{run_index + 1:05d}"
            controls = {"max_interventions": max_interventions}
            frames = run_single_scenario(
                scenario_path=scenario_path,
                steps=steps,
                seed=seed,
                run_id=run_id,
                controls=controls,
                final_metrics_only=True,
            )
            metrics = frames.get("outcome_metrics", pd.DataFrame()).copy()
            events = frames.get("events", pd.DataFrame()).copy()
            if not metrics.empty:
                metrics["dose_arm"] = arm
                metrics["seed"] = seed
                metrics_frames.append(metrics)
            if not events.empty:
                events["dose_arm"] = arm
                events["seed"] = seed
                event_frames.append(events)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metrics_all = combine_frame_list(metrics_frames)
    events_all = combine_frame_list(event_frames)
    summary = _summarize(metrics_all, events_all)
    paper_table = _paper_table(summary)
    metrics_all.to_parquet(output / "dose_limited_s2_metrics.parquet", index=False)
    events_all.to_parquet(output / "dose_limited_s2_events.parquet", index=False)
    summary.to_csv(output / "dose_limited_s2_summary.csv", index=False)
    paper_table.to_csv(output / "dose_limited_s2_paper_table.csv", index=False)
    paper_table.to_latex(output / "dose_limited_s2_paper_table.tex", index=False, escape=True)
    return paper_table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/validation")
    parser.add_argument("--scenario-path", default="config/scenarios/S2_critic_feedback.yaml")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--base-seed", type=int, default=20260700)
    args = parser.parse_args()
    table = run_dose_limited_s2(
        output_dir=args.output_dir,
        scenario_path=args.scenario_path,
        steps=args.steps,
        runs=args.runs,
        base_seed=args.base_seed,
    )
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
