from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _section_table(title: str, df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty:
        return f"## {title}\n\n_Not available._\n\n"
    rows = df.head(max_rows).astype(str)
    header = "| " + " | ".join(rows.columns) + " |"
    divider = "| " + " | ".join(["---"] * len(rows.columns)) + " |"
    body = [
        "| " + " | ".join(row.replace("\n", " ") for row in record) + " |"
        for record in rows.to_numpy()
    ]
    return f"## {title}\n\n" + "\n".join([header, divider, *body]) + "\n\n"


def build_validation_report(validation_dir: str | Path, output: str | Path) -> Path:
    root = Path(validation_dir)
    invariant = _read_csv(root / "invariant_checks.csv")
    extreme = _read_csv(root / "extreme_condition_summary_all_scenarios.csv")
    if extreme.empty:
        extreme = _read_csv(root / "extreme_condition_summary.csv")
    windows = _read_csv(root / "robustness_window_summary.csv")
    oat = _read_csv(root / "sensitivity_oat_summary_all_scenarios.csv")
    if oat.empty:
        oat = _read_csv(root / "sensitivity_oat_summary.csv")
    factorial = _read_csv(root / "sensitivity_factorial_summary.csv")

    md = ["# Stage 12 Validation, Verification, Robustness, and Sensitivity Report\n\n", "## Summary\n\n"]
    if not invariant.empty and "passed" in invariant.columns:
        md.append(f"- Invariant check pass rate: **{invariant['passed'].mean():.1%}**\n")
        md.append(f"- Failed invariant checks: **{int((~invariant['passed']).sum())}**\n")
    if not extreme.empty and "validation_scenario" in extreme.columns:
        md.append(f"- Extreme-condition scenarios evaluated: **{extreme['validation_scenario'].nunique()}**\n")
    if not oat.empty and "validation_scenario" in oat.columns:
        md.append(f"- OAT sensitivity scenarios evaluated: **{oat['validation_scenario'].nunique()}**\n")
    md.append("\nNER is treated as a net index with expected range [-1, 1].\n\n")
    md.append(_section_table("Invariant checks", invariant, 50))
    md.append(_section_table("Extreme-condition validation", extreme, 50))
    md.append(_section_table("Final-window robustness", windows, 50))
    md.append(_section_table("One-at-a-time sensitivity", oat, 50))
    md.append(_section_table("Factorial sensitivity", factorial, 30))

    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(md), encoding="utf-8")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-dir", default="outputs/validation")
    parser.add_argument("--output", default="outputs/validation/validation_report.md")
    args = parser.parse_args()
    path = build_validation_report(args.validation_dir, args.output)
    print(f"Saved validation report to {path.resolve()}")


if __name__ == "__main__":
    main()
