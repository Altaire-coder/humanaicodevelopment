import argparse
from pathlib import Path
import yaml
import numpy as np
import pandas as pd

from src.metrics.risk import recursive_degradation_risk

def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def run(config):
    rng = np.random.default_rng(config["simulation"]["seed"])
    rows = []

    for run_id in range(config["simulation"]["runs"]):
        diversity = 1.0
        performance = 0.75
        error = 0.10
        lineage = 0.10

        for t in range(config["simulation"]["steps"]):
            reuse = config["platform"]["interaction_data_reuse"]
            verification = config["human"]["verification_mean"]
            mutation = config["idea"]["mutation_rate"]

            diversity = np.clip(diversity - 0.01 * reuse + 0.008 * verification, 0, 1)
            lineage = np.clip(lineage + 0.012 * reuse - 0.006 * verification, 0, 1)
            error = np.clip(error + mutation * reuse - 0.03 * verification, 0, 1)
            performance = np.clip(
                performance + 0.01 * diversity - 0.02 * error, 0, 1
            )

            risk = recursive_degradation_risk(
                lineage_concentration=lineage,
                diversity_loss=1-diversity,
                error_persistence=error,
                error_reproduction=min(2*error, 1),
                intention_drift=1-performance,
                performance_decline=1-performance,
                human_reliance=config["human"]["ai_reliance_mean"],
                recovery_probability=diversity * verification,
                tail_survival=diversity,
            )

            rows.append({
                "run_id": run_id,
                "time": t,
                "diversity": diversity,
                "lineage_concentration": lineage,
                "error": error,
                "performance": performance,
                "risk": risk,
            })

    return pd.DataFrame(rows)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/base.yaml")
    parser.add_argument("--output", default="outputs/results.parquet")
    args = parser.parse_args()

    config = load_config(args.config)
    df = run(config)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output, index=False)
    print(f"Saved {len(df):,} rows to {output}")

if __name__ == "__main__":
    main()
