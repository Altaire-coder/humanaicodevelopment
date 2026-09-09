from __future__ import annotations

import argparse
from pathlib import Path

from src.metrics.pipeline import (
    MetricsPipelineConfig,
    compute_metrics_from_directory,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="outputs/stage6")
    parser.add_argument(
        "--output",
        default="outputs/stage6/outcome_metrics.parquet",
    )
    parser.add_argument("--window-size", type=int, default=10)
    parser.add_argument("--baseline-window", type=int, default=5)
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    result = compute_metrics_from_directory(
        input_dir=args.input_dir,
        output_path=str(output),
        config=MetricsPipelineConfig(
            window_size=args.window_size,
            baseline_window=args.baseline_window,
            show_progress=True,
        ),
    )

    print(f"Saved {len(result):,} metric rows to {output.resolve()}")

    if result.empty:
        print("No metric rows were computed.")
        return

    summary_cols = [
        "co_development_score",
        "co_degradation_score",
        "co_extinction_risk",
        "recursive_degradation_risk",
        "R_V",
        "R_E",
        "NER",
    ]
    available_cols = [col for col in summary_cols if col in result.columns]

    if available_cols:
        print(
            result.groupby("scenario_id")[available_cols]
            .mean()
            .round(3)
            .to_string()
        )


if __name__ == "__main__":
    main()
