from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
from .common import CORE_METRICS, combine_frame_list, progress_iter, run_single_scenario, save_table

OAT_GRID = {
    "human_accuracy": [0.50, 0.60, 0.70, 0.80, 0.90],
    "ai_accuracy": [0.50, 0.60, 0.70, 0.80, 0.90],
    "verification": [0.00, 0.25, 0.50, 0.75, 1.00],
    "sycophancy": [0.00, 0.25, 0.50, 0.75],
    "ai_memory_strength": [0.48, 0.60, 0.72],
    "ai_reuse_ratio": [0.25, 0.50, 0.75, 0.90],
    "context_contamination_rate": [0.024, 0.030, 0.036],
    "memory_decay": [0.040, 0.050, 0.060],
    "human_learning_rate": [0.064, 0.080, 0.096],
    "human_dependence_rate": [0.040, 0.050, 0.060],
    "mutation_rate": [0.05, 0.10, 0.20, 0.30],
    "reset_threshold": [0.015, 0.020, 0.025, 0.030],
    "context_reset_strength": [0.30, 0.50, 0.70],
}


def summarize_oat(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    final = metrics.sort_values("time").groupby(["validation_scenario", "parameter", "parameter_value", "run_id"], as_index=False).tail(1)
    available = [m for m in CORE_METRICS if m in final.columns]
    rows = []
    for (scenario, parameter, value), group in final.groupby(["validation_scenario", "parameter", "parameter_value"]):
        for metric in available:
            rows.append({"validation_scenario": scenario, "parameter": parameter, "parameter_value": value, "metric": metric, "mean": group[metric].mean(), "std": group[metric].std(), "count": group[metric].count()})
    return pd.DataFrame(rows)


def run_oat_sensitivity(*, scenario_path: str, output_dir: str | Path, parameters: list[str] | None = None, steps: int = 100, runs: int = 20, base_seed: int = 8000, scenario_label: str | None = None, show_progress: bool = True) -> pd.DataFrame:
    parameters = parameters or list(OAT_GRID.keys())
    label = scenario_label or Path(scenario_path).stem
    tasks = []
    for parameter in parameters:
        if parameter not in OAT_GRID:
            raise ValueError(f"Unknown OAT parameter: {parameter}")
        for value in OAT_GRID[parameter]:
            for run_index in range(runs):
                tasks.append((parameter, value, run_index))
    all_metrics = []
    for parameter, value, run_index in progress_iter(tasks, total=len(tasks), desc=f"OAT {label}", disable=not show_progress):
        seed = base_seed + run_index
        frames = run_single_scenario(
            scenario_path=scenario_path,
            steps=steps,
            seed=seed,
            run_id=f"{label}_{parameter}_{value}_R{run_index + 1:05d}",
            controls={parameter: value},
            final_metrics_only=True,
        )
        metrics = frames.get("outcome_metrics", pd.DataFrame())
        if metrics.empty:
            continue
        metrics = metrics.copy()
        metrics["validation_scenario"] = label
        metrics["parameter"] = parameter
        metrics["parameter_value"] = value
        metrics["seed"] = seed
        all_metrics.append(metrics)
    metrics_all = combine_frame_list(all_metrics)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not metrics_all.empty:
        metrics_all.to_parquet(out_dir / f"sensitivity_oat_metrics_{label}.parquet", index=False)
    summary = summarize_oat(metrics_all)
    save_table(summary, out_dir / f"sensitivity_oat_summary_{label}.csv")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario-path", default="config/scenarios/S0_no_treatment.yaml")
    parser.add_argument("--scenario-label", default=None)
    parser.add_argument("--output-dir", default="outputs/validation")
    parser.add_argument("--parameters", nargs="*", default=None)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--base-seed", type=int, default=8000)
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()
    out = run_oat_sensitivity(scenario_path=args.scenario_path, scenario_label=args.scenario_label, output_dir=args.output_dir, parameters=args.parameters, steps=args.steps, runs=args.runs, base_seed=args.base_seed, show_progress=not args.no_progress)
    print(out.head(50).to_string(index=False))


if __name__ == "__main__":
    main()
