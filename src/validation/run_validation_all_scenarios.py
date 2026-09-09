from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
from .common import DEFAULT_SCENARIOS, combine_frame_list, progress_iter, save_table
from .export_validation_report import build_validation_report
from .extreme_conditions import run_extreme_conditions
from .robustness_windows import robustness_across_windows
from .sensitivity_oat import run_oat_sensitivity
from .verify_invariants import verify_stage_outputs


def _combine_csvs(output_dir: Path, pattern: str, combined_name: str) -> None:
    frames = []
    for path in sorted(output_dir.glob(pattern)):
        if path.name == combined_name:
            continue
        try:
            frames.append(pd.read_csv(path))
        except Exception:
            pass
    combined = combine_frame_list(frames)
    if not combined.empty:
        save_table(combined, output_dir / combined_name)


def parse_scenario_subset(values: list[str] | None) -> dict[str, str]:
    if not values:
        return DEFAULT_SCENARIOS.copy()
    selected = {}
    for value in values:
        if value in DEFAULT_SCENARIOS:
            selected[value] = DEFAULT_SCENARIOS[value]
        else:
            path = Path(value)
            selected[path.stem] = str(path)
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-dir", default="outputs/stage6")
    parser.add_argument("--output-dir", default="outputs/validation")
    parser.add_argument("--scenarios", nargs="*", default=None)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--oat-parameters", nargs="*", default=["verification", "sycophancy", "ai_reuse_ratio", "reset_threshold"])
    parser.add_argument("--skip-extreme", action="store_true")
    parser.add_argument("--skip-oat", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    save_table(verify_stage_outputs(args.stage_dir), out / "invariant_checks.csv")
    robustness_across_windows(outcome_path=Path(args.stage_dir) / "outcome_metrics.parquet", output=out / "robustness_window_summary.csv", windows=[10, 20, 30])

    scenarios = parse_scenario_subset(args.scenarios)
    for scenario_label, scenario_path in progress_iter(list(scenarios.items()), total=len(scenarios), desc="validation scenarios", disable=args.no_progress):
        if not args.skip_extreme:
            run_extreme_conditions(scenario_path=scenario_path, scenario_label=scenario_label, output_dir=out, steps=args.steps, runs=args.runs, base_seed=7000, show_progress=not args.no_progress)
        if not args.skip_oat:
            run_oat_sensitivity(scenario_path=scenario_path, scenario_label=scenario_label, output_dir=out, parameters=args.oat_parameters, steps=args.steps, runs=args.runs, base_seed=8000, show_progress=not args.no_progress)

    _combine_csvs(out, "extreme_condition_summary_*.csv", "extreme_condition_summary_all_scenarios.csv")
    _combine_csvs(out, "sensitivity_oat_summary_*.csv", "sensitivity_oat_summary_all_scenarios.csv")
    build_validation_report(validation_dir=out, output=out / "validation_report.md")
    print(f"Saved multi-scenario Stage 12 validation outputs to {out.resolve()}")


if __name__ == "__main__":
    main()
