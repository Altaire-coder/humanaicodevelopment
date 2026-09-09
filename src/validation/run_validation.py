from __future__ import annotations

import argparse
from pathlib import Path

from .export_validation_report import build_validation_report
from .extreme_conditions import run_extreme_conditions
from .robustness_windows import robustness_across_windows
from .sensitivity_oat import run_oat_sensitivity
from .verify_invariants import verify_stage_outputs
from .common import save_table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-dir", default="outputs/stage6")
    parser.add_argument("--output-dir", default="outputs/validation")
    parser.add_argument("--scenario-path", default="config/scenarios/S0_no_treatment.yaml")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--skip-extreme", action="store_true")
    parser.add_argument("--skip-oat", action="store_true")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    invariant = verify_stage_outputs(args.stage_dir)
    save_table(invariant, out / "invariant_checks.csv")

    robustness_across_windows(
        outcome_path=Path(args.stage_dir) / "outcome_metrics.parquet",
        output=out / "robustness_window_summary.csv",
        windows=[10, 20, 30],
    )

    if not args.skip_extreme:
        run_extreme_conditions(
            scenario_path=args.scenario_path,
            output_dir=out,
            steps=args.steps,
            runs=args.runs,
        )

    if not args.skip_oat:
        run_oat_sensitivity(
            scenario_path=args.scenario_path,
            output_dir=out,
            parameters=["verification", "sycophancy", "ai_reuse_ratio", "reset_threshold"],
            steps=args.steps,
            runs=args.runs,
        )

    build_validation_report(
        validation_dir=out,
        output=out / "validation_report.md",
    )

    print(f"Saved Stage 12 validation outputs to {out.resolve()}")


if __name__ == "__main__":
    main()
