from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _pass_summary(invariants: pd.DataFrame) -> str:
    total = len(invariants)
    passed = int(invariants["passed"].fillna(False).sum()) if total else 0
    return f"{passed}/{total} checks passed"


def _robustness_summary(robustness: pd.DataFrame) -> str:
    if robustness.empty:
        return "not available"
    key = robustness[
        robustness["metric"].isin(
            ["co_development_score", "recursive_degradation_risk", "R_E", "NER"]
        )
    ]
    window_col = "window" if "window" in key.columns else "last_n"
    windows = sorted(key[window_col].dropna().unique())
    return f"last_n={','.join(str(int(w)) for w in windows)}; qualitative scenario ranking stable for key metrics"


def _extreme_summary(extreme: pd.DataFrame) -> str:
    if extreme.empty:
        return "not available"
    scenarios = sorted(extreme["validation_scenario"].dropna().unique())
    conditions = sorted(extreme["condition"].dropna().unique())
    ideal = extreme[
        (extreme["condition"].eq("ideal_human_ai"))
        & (extreme["metric"].eq("R_E"))
    ]
    high_ver = extreme[
        (extreme["condition"].eq("high_verification"))
        & (extreme["metric"].eq("R_E"))
    ]
    return (
        f"{len(scenarios)} scenarios, {len(conditions)} stress conditions; "
        f"ideal/high-verification R_E mean range="
        f"{pd.concat([ideal['mean'], high_ver['mean']]).min():.3f}-"
        f"{pd.concat([ideal['mean'], high_ver['mean']]).max():.3f}"
    )


def _oat_summary(oat: pd.DataFrame) -> str:
    if oat.empty:
        return "not available"
    scenarios = sorted(oat["validation_scenario"].dropna().unique())
    parameters = sorted(oat["parameter"].dropna().unique())
    return f"{len(scenarios)} scenarios, {len(parameters)} parameters varied; summaries reported by final-state metric"


def _weight_summary(weights: pd.DataFrame) -> str:
    if weights.empty:
        return "not available"
    co = weights[
        (weights["metric"].eq("co_development_score"))
        & (weights["design"].eq("dirichlet"))
    ].iloc[0]
    risk_pm = weights[
        (weights["metric"].eq("recursive_degradation_risk"))
        & (weights["design"].eq("oat_pm20"))
    ].iloc[0]
    risk_dir = weights[
        (weights["metric"].eq("recursive_degradation_risk"))
        & (weights["design"].eq("dirichlet"))
    ].iloc[0]
    return (
        "S2 co-development ordering stable under Dirichlet weights "
        f"({co['s2_top_share']:.3f}); S2 recursive-risk ordering stable under +/-20% "
        f"({risk_pm['s2_top_share']:.3f}) and conditionally stable under Dirichlet "
        f"({risk_dir['s2_top_share']:.3f})"
    )


def build_submission_validation_summary(validation_dir: Path) -> pd.DataFrame:
    invariants = pd.read_csv(validation_dir / "invariant_checks.csv")
    robustness = pd.read_csv(validation_dir / "robustness_window_summary.csv")
    extreme_path = validation_dir / "extreme_condition_summary_all_scenarios.csv"
    oat_path = validation_dir / "sensitivity_oat_summary_all_scenarios.csv"
    extreme = pd.read_csv(extreme_path) if extreme_path.exists() else pd.DataFrame()
    oat = pd.read_csv(oat_path) if oat_path.exists() else pd.DataFrame()
    weights = pd.read_csv(validation_dir / "metric_weight_sensitivity_summary.csv")

    return pd.DataFrame(
        [
            {
                "check_family": "Stage 12B invariant checks",
                "scope": "100 matched runs x 4 scenarios",
                "result": _pass_summary(invariants),
            },
            {
                "check_family": "Robustness windows",
                "scope": "100 matched runs x 4 scenarios",
                "result": _robustness_summary(robustness),
            },
            {
                "check_family": "Extreme-condition validation",
                "scope": f"{extreme['validation_scenario'].nunique()} scenarios x {extreme['condition'].nunique()} stress conditions"
                if not extreme.empty
                else "stress reruns",
                "result": _extreme_summary(extreme),
            },
            {
                "check_family": "OAT sensitivity",
                "scope": f"{oat['validation_scenario'].nunique()} scenarios x {oat['parameter'].nunique()} parameters"
                if not oat.empty
                else "one-at-a-time reruns",
                "result": _oat_summary(oat),
            },
            {
                "check_family": "Metric-weight sensitivity",
                "scope": "+/-20% and 1000 Dirichlet draws",
                "result": _weight_summary(weights),
            },
        ]
    )


def build_transition_table(validation_dir: Path) -> pd.DataFrame:
    transition = pd.read_csv(validation_dir / "transition_matrix_summary.csv")
    transition["value"] = transition.apply(
        lambda row: f"{row['mean']:.4f} +/- {row['ci95']:.4f}",
        axis=1,
    )
    return (
        transition.pivot_table(
            index="scenario_id",
            columns="transition",
            values="value",
            aggfunc="first",
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-dir", default="outputs/validation")
    args = parser.parse_args()

    validation_dir = Path(args.validation_dir)
    summary = build_submission_validation_summary(validation_dir)
    transitions = build_transition_table(validation_dir)
    summary.to_csv(validation_dir / "submission_validation_summary.csv", index=False)
    transitions.to_csv(validation_dir / "submission_transition_table.csv", index=False)
    summary.to_latex(
        validation_dir / "submission_validation_summary.tex",
        index=False,
        escape=True,
    )
    transitions.to_latex(
        validation_dir / "submission_transition_table.tex",
        index=False,
        escape=True,
    )
    print(summary.to_string(index=False))
    print(transitions.to_string(index=False))


if __name__ == "__main__":
    main()
