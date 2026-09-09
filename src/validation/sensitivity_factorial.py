from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import pandas as pd

from .common import CORE_METRICS, combine_frame_list, run_single_scenario, save_table


FACTORIAL_LEVELS = {
    "human_accuracy": [0.60, 0.85],
    "ai_accuracy": [0.60, 0.85],
    "verification": [0.25, 0.75],
    "sycophancy": [0.10, 0.70],
    "ai_reuse_ratio": [0.50, 0.90],
}


def build_factorial_design(parameters: list[str] | None = None) -> list[dict]:
    parameters = parameters or list(FACTORIAL_LEVELS.keys())
    levels = [FACTORIAL_LEVELS[p] for p in parameters]
    design = []
    for values in itertools.product(*levels):
        design.append(dict(zip(parameters, values)))
    return design


def run_factorial_sensitivity(
    *,
    scenario_path: str,
    output_dir: str | Path,
    parameters: list[str] | None = None,
    steps: int = 100,
    runs: int = 20,
    base_seed: int = 9000,
) -> pd.DataFrame:
    design = build_factorial_design(parameters)
    all_metrics = []

    for setting_index, controls in enumerate(design):
        setting_id = f"F{setting_index + 1:03d}"
        for run_index in range(runs):
            seed = base_seed + run_index
            frames = run_single_scenario(
                scenario_path=scenario_path,
                steps=steps,
                seed=seed,
                run_id=f"{setting_id}_R{run_index + 1:05d}",
                controls=controls,
            )
            metrics = frames.get("outcome_metrics", pd.DataFrame())
            if metrics.empty:
                continue
            metrics = metrics.copy()
            metrics["setting_id"] = setting_id
            metrics["seed"] = seed
            for key, value in controls.items():
                metrics[key] = value
            all_metrics.append(metrics)

    metrics_all = combine_frame_list(all_metrics)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not metrics_all.empty:
        metrics_all.to_parquet(out_dir / "sensitivity_factorial_metrics.parquet", index=False)

    summary = summarize_factorial(metrics_all, parameters or list(FACTORIAL_LEVELS.keys()))
    save_table(summary, out_dir / "sensitivity_factorial_summary.csv")
    return summary


def summarize_factorial(metrics: pd.DataFrame, parameters: list[str]) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()

    final = (
        metrics.sort_values("time")
        .groupby(["setting_id", "run_id"], as_index=False)
        .tail(1)
    )
    available = [m for m in CORE_METRICS if m in final.columns]

    rows = []
    group_cols = ["setting_id"] + parameters
    for keys, group in final.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row_base = dict(zip(group_cols, keys))
        for metric in available:
            row = row_base.copy()
            row.update(
                {
                    "metric": metric,
                    "mean": group[metric].mean(),
                    "std": group[metric].std(),
                    "count": group[metric].count(),
                }
            )
            rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario-path", default="config/scenarios/S0_no_treatment.yaml")
    parser.add_argument("--output-dir", default="outputs/validation")
    parser.add_argument("--parameters", nargs="*", default=None)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--base-seed", type=int, default=9000)
    args = parser.parse_args()

    out = run_factorial_sensitivity(
        scenario_path=args.scenario_path,
        output_dir=args.output_dir,
        parameters=args.parameters,
        steps=args.steps,
        runs=args.runs,
        base_seed=args.base_seed,
    )
    print(out.head(50).to_string(index=False))
    print(f"Saved factorial sensitivity results to {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
