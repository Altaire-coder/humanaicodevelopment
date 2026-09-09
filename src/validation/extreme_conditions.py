from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
from .common import CORE_METRICS, combine_frame_list, progress_iter, run_single_scenario, save_table

EXTREME_CONDITIONS = {
    "ideal_human_ai": {"human_accuracy": 1.0, "ai_accuracy": 1.0, "verification": 1.0, "sycophancy": 0.0, "ai_reuse_ratio": 0.50},
    "poor_human_ai": {"human_accuracy": 0.0, "ai_accuracy": 0.0, "verification": 0.0, "sycophancy": 0.80, "ai_reuse_ratio": 0.90},
    "high_verification": {"verification": 1.0, "sycophancy": 0.0},
    "no_verification": {"verification": 0.0, "sycophancy": 0.50},
    "high_sycophancy": {"sycophancy": 0.90, "verification": 0.25},
    "low_sycophancy": {"sycophancy": 0.00, "verification": 0.75},
}


def summarize_extreme_conditions(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    final = metrics.sort_values("time").groupby(["validation_scenario", "condition", "run_id"], as_index=False).tail(1)
    available = [m for m in CORE_METRICS if m in final.columns]
    rows = []
    for (scenario, condition), group in final.groupby(["validation_scenario", "condition"]):
        for metric in available:
            rows.append({"validation_scenario": scenario, "condition": condition, "metric": metric, "mean": group[metric].mean(), "std": group[metric].std(), "count": group[metric].count()})
    return pd.DataFrame(rows)


def run_extreme_conditions(*, scenario_path: str, output_dir: str | Path, steps: int = 100, runs: int = 20, base_seed: int = 7000, scenario_label: str | None = None, show_progress: bool = True) -> pd.DataFrame:
    label = scenario_label or Path(scenario_path).stem
    tasks = [(name, controls, i) for name, controls in EXTREME_CONDITIONS.items() for i in range(runs)]
    all_metrics = []
    for condition_name, controls, run_index in progress_iter(tasks, total=len(tasks), desc=f"extreme {label}", disable=not show_progress):
        seed = base_seed + run_index
        frames = run_single_scenario(
            scenario_path=scenario_path,
            steps=steps,
            seed=seed,
            run_id=f"{label}_{condition_name}_R{run_index + 1:05d}",
            controls=controls,
            final_metrics_only=True,
        )
        metrics = frames.get("outcome_metrics", pd.DataFrame())
        if metrics.empty:
            continue
        metrics = metrics.copy()
        metrics["validation_scenario"] = label
        metrics["condition"] = condition_name
        metrics["seed"] = seed
        all_metrics.append(metrics)

    metrics_all = combine_frame_list(all_metrics)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not metrics_all.empty:
        metrics_all.to_parquet(out_dir / f"extreme_condition_metrics_{label}.parquet", index=False)
    summary = summarize_extreme_conditions(metrics_all)
    save_table(summary, out_dir / f"extreme_condition_summary_{label}.csv")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario-path", default="config/scenarios/S0_no_treatment.yaml")
    parser.add_argument("--scenario-label", default=None)
    parser.add_argument("--output-dir", default="outputs/validation")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--base-seed", type=int, default=7000)
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()
    out = run_extreme_conditions(scenario_path=args.scenario_path, scenario_label=args.scenario_label, output_dir=args.output_dir, steps=args.steps, runs=args.runs, base_seed=args.base_seed, show_progress=not args.no_progress)
    print(out.head(50).to_string(index=False))


if __name__ == "__main__":
    main()
