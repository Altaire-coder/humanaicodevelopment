from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
from .common import read_stage_outputs, save_table

BOUNDED_0_1_COLUMNS = [
    "validity", "novelty", "alignment", "error_severity",
    "human_confidence", "ai_confidence", "context_contamination",
    "model_contamination", "co_development_score", "co_degradation_score",
    "co_extinction_risk", "recursive_degradation_risk",
    "R_V", "R_E", "R_N", "recovery_probability",
]
BOUNDED_MINUS1_1_COLUMNS = ["NER"]


def _record(check: str, passed: bool, details: str = "") -> dict:
    return {"check": check, "passed": bool(passed), "details": details}


def _check_range(rows, table_name, frame, col, lower, upper):
    if col not in frame.columns:
        return
    values = frame[col].dropna()
    if values.empty:
        return
    rows.append(_record(
        f"{table_name}.{col} within [{lower:g}, {upper:g}]",
        bool(((values >= lower) & (values <= upper)).all()),
        f"min={values.min():.6f}, max={values.max():.6f}",
    ))


def verify_stage_outputs(input_dir: str | Path) -> pd.DataFrame:
    frames = read_stage_outputs(input_dir)
    rows = []
    required = ["events", "idea_states", "human_states", "ai_states", "local_ai_states", "platform_states", "lineage_edges", "outcome_metrics"]
    for name in required:
        frame = frames.get(name, pd.DataFrame())
        rows.append(_record(f"required table exists and nonempty: {name}", not frame.empty, f"rows={len(frame)}"))

    events = frames.get("events", pd.DataFrame())
    metrics = frames.get("outcome_metrics", pd.DataFrame())

    if not events.empty:
        required_event_cols = {"scenario_id", "run_id", "time", "event_type"}
        rows.append(_record("events contain required columns", required_event_cols.issubset(events.columns), f"missing={sorted(required_event_cols - set(events.columns))}"))

        if {"scenario_id", "run_id", "matched_run_id", "seed"}.issubset(events.columns):
            inventory = events[["scenario_id", "run_id", "matched_run_id", "seed"]].drop_duplicates().groupby("scenario_id").agg(
                runs=("run_id", "nunique"), matched_runs=("matched_run_id", "nunique"), min_seed=("seed", "min"), max_seed=("seed", "max")
            )
            balanced = inventory["runs"].nunique() == 1 and inventory["matched_runs"].nunique() == 1 and inventory["min_seed"].nunique() == 1 and inventory["max_seed"].nunique() == 1
            rows.append(_record("matched-seed scenario balance", balanced, repr(inventory.to_dict(orient="index"))))

        if "is_intervention" in events.columns:
            s0 = events[events["scenario_id"].eq("S0_no_treatment")]
            rows.append(_record("S0 has no intervention events", s0.empty or not s0["is_intervention"].fillna(False).any(), f"S0_interventions={int(s0['is_intervention'].fillna(False).sum()) if not s0.empty else 0}"))
            for scenario_id, expected_type in [
                ("S1_human_data_injection", "human_data_injection"),
                ("S2_critic_feedback", "critic_feedback"),
                ("S3_context_reset", "context_reset"),
            ]:
                sub = events[events["scenario_id"].eq(scenario_id) & events["is_intervention"].fillna(False)]
                if sub.empty:
                    passed, details = False, "no intervention events"
                else:
                    passed, details = sub["event_type"].eq(expected_type).all(), repr(sub["event_type"].value_counts().to_dict())
                rows.append(_record(f"{scenario_id} intervention type is {expected_type}", passed, details))

    for table_name, frame in frames.items():
        if frame.empty:
            continue
        for col in BOUNDED_0_1_COLUMNS:
            _check_range(rows, table_name, frame, col, 0.0, 1.0)
        for col in BOUNDED_MINUS1_1_COLUMNS:
            _check_range(rows, table_name, frame, col, -1.0, 1.0)

    if not metrics.empty:
        final = metrics.sort_values("time").groupby(["scenario_id", "run_id"]).tail(1)
        rows.append(_record("final metric row for every scenario-run", len(final) == metrics[["scenario_id", "run_id"]].drop_duplicates().shape[0], f"final_rows={len(final)}"))

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="outputs/stage6")
    parser.add_argument("--output", default="outputs/validation/invariant_checks.csv")
    args = parser.parse_args()
    out = verify_stage_outputs(args.input_dir)
    save_table(out, args.output)
    print(out.to_string(index=False))
    print(f"Saved invariant checks to {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
