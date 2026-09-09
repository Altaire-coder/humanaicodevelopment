from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run_command(command: list[str]) -> None:
    print("+ " + " ".join(command))
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the default matched Monte Carlo simulation workflow."
    )
    parser.add_argument(
        "--config",
        default="config/scenario_experiment.yaml",
        help="Experiment configuration file.",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/results",
        help="Directory where simulation, analysis, figures, and tables are written.",
    )
    parser.add_argument(
        "--run-start",
        type=int,
        default=1,
        help="One-based matched run index to start from.",
    )
    parser.add_argument(
        "--run-count",
        type=int,
        default=100,
        help="Number of matched Monte Carlo runs to execute.",
    )
    parser.add_argument(
        "--burn-in",
        type=int,
        default=5,
        help="Burn-in period used for trajectory figures.",
    )
    parser.add_argument(
        "--vector-format",
        choices=["pdf", "svg"],
        default="pdf",
        help="Vector format for generated figures.",
    )
    args = parser.parse_args()
    os.environ.setdefault("MPLBACKEND", "Agg")

    output_root = Path(args.output_root)
    simulation_dir = output_root / "simulation"
    analysis_dir = output_root / "analysis"

    run_command(
        [
            sys.executable,
            "-m",
            "src.simulation.run_scenarios_batch",
            "--config",
            args.config,
            "--run-start",
            str(args.run_start),
            "--run-count",
            str(args.run_count),
            "--output-dir",
            str(simulation_dir),
        ]
    )
    run_command(
        [
            sys.executable,
            "-m",
            "src.analysis.run_analysis",
            "--input-dir",
            str(simulation_dir),
            "--output-dir",
            str(analysis_dir),
        ]
    )
    run_command(
        [
            sys.executable,
            "-m",
            "src.analysis.transition_matrix",
            "--events",
            str(simulation_dir / "events.parquet"),
            "--output-dir",
            str(analysis_dir),
        ]
    )
    run_command(
        [
            sys.executable,
            "-m",
            "src.visualization.simulation_figures",
            "--input-dir",
            str(simulation_dir),
            "--analysis-dir",
            str(analysis_dir),
            "--output-dir",
            str(output_root),
            "--burn-in",
            str(args.burn_in),
            "--vector-format",
            args.vector_format,
        ]
    )

    print(f"Default simulation workflow complete: {output_root.resolve()}")


if __name__ == "__main__":
    main()
