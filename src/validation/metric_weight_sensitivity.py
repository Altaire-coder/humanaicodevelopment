from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


RISK_COMPONENTS = [
    "lineage_concentration",
    "diversity_loss",
    "error_persistence",
    "error_activity_component",
    "intention_drift",
    "performance_decline",
    "human_ai_reliance",
    "inverse_recovery",
    "inverse_tail_survival",
]


def _clip01(values):
    return np.clip(values, 0.0, 1.0)


def _normalize(weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    total = weights.sum()
    if total <= 0:
        return np.ones_like(weights) / len(weights)
    return weights / total


def prepare_final_components(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    ordered = metrics.sort_values("time").copy()
    initial = (
        ordered.groupby(["scenario_id", "run_id"], as_index=False)
        .first()[
            [
                "scenario_id",
                "run_id",
                "mean_validity",
                "mean_novelty",
                "mean_alignment",
            ]
        ]
        .rename(
            columns={
                "mean_validity": "initial_validity",
                "mean_novelty": "initial_novelty",
                "mean_alignment": "initial_alignment",
            }
        )
    )
    final = ordered.groupby(["scenario_id", "run_id"], as_index=False).tail(1)
    final = final.merge(initial, on=["scenario_id", "run_id"], how="left")
    final["diversity_loss"] = _clip01(final["initial_novelty"] - final["mean_novelty"])
    final["error_persistence"] = _clip01(final["mean_error_severity"])
    final["error_activity_component"] = _clip01(final["R_E"] / (1.0 + final["R_E"]))
    final["intention_drift"] = _clip01(final["initial_alignment"] - final["mean_alignment"])
    final["performance_decline"] = _clip01(final["initial_validity"] - final["mean_validity"])
    final["inverse_recovery"] = _clip01(1.0 - final["recovery_probability"])
    final["inverse_tail_survival"] = _clip01(1.0 - final["tail_idea_survival"])
    return final


def weighted_co_development(final: pd.DataFrame, weights: np.ndarray) -> np.ndarray:
    weights = _normalize(weights)
    preservation = _clip01(final["preservation"].to_numpy(dtype=float))
    expansion = _clip01(final["productive_expansion"].to_numpy(dtype=float))
    return np.exp(weights[0] * np.log(np.clip(preservation, 1e-12, 1.0)) + weights[1] * np.log(np.clip(expansion, 1e-12, 1.0)))


def weighted_recursive_risk(final: pd.DataFrame, weights: np.ndarray) -> np.ndarray:
    weights = _normalize(weights)
    values = final[RISK_COMPONENTS].to_numpy(dtype=float)
    return values @ weights


def summarize_weight_settings(
    final: pd.DataFrame,
    *,
    metric: str,
    values: np.ndarray,
    design: str,
    setting_id: str,
) -> dict:
    work = final[["scenario_id", "run_id"]].copy()
    work["value"] = values
    means = work.groupby("scenario_id")["value"].mean()
    s0 = float(means.get("S0_no_treatment", np.nan))
    s2 = float(means.get("S2_critic_feedback", np.nan))
    top = str(means.idxmax())
    return {
        "design": design,
        "setting_id": setting_id,
        "metric": metric,
        "S0_no_treatment": s0,
        "S1_human_data_injection": float(means.get("S1_human_data_injection", np.nan)),
        "S2_critic_feedback": s2,
        "S3_context_reset": float(means.get("S3_context_reset", np.nan)),
        "S2_minus_S0": s2 - s0,
        "top_scenario": top,
        "S2_is_top": top == "S2_critic_feedback",
        "S2_gt_S0": s2 > s0,
    }


def build_weight_sensitivity(
    metrics: pd.DataFrame,
    *,
    dirichlet_draws: int = 1000,
    seed: int = 12345,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    final = prepare_final_components(metrics)
    rng = np.random.default_rng(seed)
    rows = []

    co_base = np.array([1.0, 1.0])
    risk_base = np.ones(len(RISK_COMPONENTS))
    for index, factor in enumerate([0.8, 1.2], start=1):
        for pos in range(len(co_base)):
            weights = co_base.copy()
            weights[pos] *= factor
            rows.append(
                summarize_weight_settings(
                    final,
                    metric="co_development_score",
                    values=weighted_co_development(final, weights),
                    design="oat_pm20",
                    setting_id=f"co_w{pos + 1}_{factor:.1f}",
                )
            )
        for pos in range(len(risk_base)):
            weights = risk_base.copy()
            weights[pos] *= factor
            rows.append(
                summarize_weight_settings(
                    final,
                    metric="recursive_degradation_risk",
                    values=weighted_recursive_risk(final, weights),
                    design="oat_pm20",
                    setting_id=f"risk_w{pos + 1}_{factor:.1f}",
                )
            )

    for draw in range(dirichlet_draws):
        co_weights = rng.dirichlet(np.ones(len(co_base)))
        risk_weights = rng.dirichlet(np.ones(len(risk_base)))
        rows.append(
            summarize_weight_settings(
                final,
                metric="co_development_score",
                values=weighted_co_development(final, co_weights),
                design="dirichlet",
                setting_id=f"co_D{draw + 1:04d}",
            )
        )
        rows.append(
            summarize_weight_settings(
                final,
                metric="recursive_degradation_risk",
                values=weighted_recursive_risk(final, risk_weights),
                design="dirichlet",
                setting_id=f"risk_D{draw + 1:04d}",
            )
        )

    setting_summary = pd.DataFrame(rows)
    aggregate = (
        setting_summary.groupby(["design", "metric"], as_index=False)
        .agg(
            settings=("setting_id", "count"),
            mean_s2_minus_s0=("S2_minus_S0", "mean"),
            min_s2_minus_s0=("S2_minus_S0", "min"),
            max_s2_minus_s0=("S2_minus_S0", "max"),
            s2_top_share=("S2_is_top", "mean"),
            s2_gt_s0_share=("S2_gt_S0", "mean"),
        )
    )
    return setting_summary, aggregate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", default="outputs/stage6/outcome_metrics.parquet")
    parser.add_argument("--output-dir", default="outputs/validation")
    parser.add_argument("--dirichlet-draws", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_parquet(args.metrics)
    setting_summary, aggregate = build_weight_sensitivity(
        metrics,
        dirichlet_draws=args.dirichlet_draws,
        seed=args.seed,
    )
    setting_summary.to_csv(output_dir / "metric_weight_sensitivity_settings.csv", index=False)
    aggregate.to_csv(output_dir / "metric_weight_sensitivity_summary.csv", index=False)
    aggregate.to_latex(
        output_dir / "metric_weight_sensitivity_summary.tex",
        index=False,
        float_format="%.4f",
    )
    print(aggregate.to_string(index=False))


if __name__ == "__main__":
    main()
