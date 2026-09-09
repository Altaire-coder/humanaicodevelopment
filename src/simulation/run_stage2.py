from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from src.environment import RecursiveModelConfig
from src.simulation import MonteCarloConfig, run_monte_carlo


def _load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/stage2.yaml")
    args = parser.parse_args()

    raw = _load_yaml(args.config)
    model_config = RecursiveModelConfig(**raw["model"])
    mc_config = MonteCarloConfig(**raw["monte_carlo"])

    outputs = run_monte_carlo(model_config, mc_config)
    print("Stage 2 Monte Carlo simulation completed.")
    for name, frame in outputs.items():
        print(f"{name}: {len(frame):,} rows")
    print(f"Saved to: {Path(mc_config.output_dir).resolve()}")


if __name__ == "__main__":
    main()
