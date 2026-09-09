# Human-AI Codevelopment Simulation

Version: v0.1 research package

This repository contains a simulation package for studying recursive human-AI feedback loops. The model represents a human agent, an AI agent, local interaction memory, platform-level state, idea genealogy, and intervention policies. It is designed to examine when repeated interaction supports productive co-development and when the same recursive structure amplifies error, dependence, context contamination, or epistemic degradation.

The current v0.1 release supports a 100-run matched Monte Carlo simulation analysis by default. It was developed alongside the study:

> From Co-Development to Recursive Degradation: A Lineage-Aware Simulation of Error Propagation in Human-AI Feedback Loops

## What This Package Does

- Simulates recursive human-AI interaction over repeated time steps.
- Tracks idea validity, novelty, alignment, error severity, AI-origin reuse, and lineage relationships.
- Compares no-treatment, human-data injection, critic feedback, and context reset scenarios.
- Computes co-development, recursive degradation risk, lineage reproduction rates, epistemic transitions, and net epistemic reproduction.
- Generates reusable simulation tables, analysis tables, validation summaries, and figures.

## Repository Layout

```text
config/                 Experiment and scenario configuration files
examples/               Small runnable demonstrations
scripts/                Reproduction entry points
src/                    Simulation, analysis, metrics, validation, and visualization code
tests/                  Unit and smoke tests
```

The public research package intentionally excludes manuscript drafts, document notes, local notebooks, generated outputs, virtual environments, caches, and machine-specific temporary files.

## Installation

Use Python 3.10 or later. A virtual environment is recommended.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

On macOS or Linux, activate the environment with:

```bash
source .venv/bin/activate
```

## Default Simulation Analysis

Run the full 100-run matched Monte Carlo workflow:

```bash
python scripts/default_simulation.py
```

By default this command:

1. runs default scenarios S0-S3 for 100 matched seeds,
2. writes simulation tables to `outputs/results/simulation`,
3. computes analysis tables in `outputs/results/analysis`,
4. builds figures in `outputs/results/figures`,
5. builds reusable CSV/LaTeX tables in `outputs/results/tables`.

The same workflow can be run stepwise:

```bash
python -m src.simulation.run_scenarios_batch --config config/scenario_experiment.yaml --run-start 1 --run-count 100 --output-dir outputs/results/simulation
python -m src.analysis.run_analysis --input-dir outputs/results/simulation --output-dir outputs/results/analysis
python -m src.analysis.transition_matrix --events outputs/results/simulation/events.parquet --output-dir outputs/results/analysis
python -m src.visualization.simulation_figures --input-dir outputs/results/simulation --analysis-dir outputs/results/analysis --output-dir outputs/results
```

## Key Outputs

- `outputs/results/simulation/outcome_metrics.parquet`: time-step metrics for each scenario and matched run.
- `outputs/results/analysis/final_state_summary.csv`: final-state scenario summaries.
- `outputs/results/analysis/scenario_contrasts.csv`: matched contrasts relative to S0.
- `outputs/results/analysis/trajectory_summary.csv`: time-resolved scenario trajectories.
- `outputs/results/analysis/transition_matrix_summary.csv`: parent-child epistemic transition estimates.
- `outputs/results/figures/`: figures in PNG and PDF or SVG formats.
- `outputs/results/tables/`: reusable CSV and LaTeX summary tables.

## Scenarios Included in v0.1

- `S0_no_treatment`: baseline recursive interaction.
- `S1_human_data_injection`: one-shot independent human-origin information injection.
- `S2_critic_feedback`: recurring critical feedback and verification pressure.
- `S3_context_reset`: threshold-triggered local context reset.

Additional experimental scenario files are included for extension studies. They do not change the default v0.1 S0-S3 workflow.

## Tests

Run the test suite with:

```bash
pytest
```

For a quick public-package check:

```bash
pytest tests/test_metrics.py tests/test_scenarios.py tests/test_metrics_pipeline.py
```

## Scope and Interpretation

This package is a stylized computational model, not an empirically calibrated estimate of a specific deployed AI system. Numerical results should be interpreted as internally comparable simulation outcomes under stated assumptions. The package is intended to support reproducible analysis, sensitivity extensions, and theory-building around recursive human-AI interaction.

## Citation

If you use this package, please cite the repository metadata in `CITATION.cff` and the associated paper.

## License

This project is released under the MIT License. See `LICENSE`.
