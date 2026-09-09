from __future__ import annotations

from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError


STAGE_TABLES = (
    "events",
    "idea_states",
    "human_states",
    "ai_states",
    "local_ai_states",
    "platform_states",
    "platform_update_events",
    "lineage_edges",
    "outcome_metrics",
)

ANALYSIS_TABLES = (
    "final_state_summary",
    "last_window_summary",
    "scenario_contrasts",
    "trajectory_summary",
    "trajectory_auc",
    "change_points_recursive_risk",
    "time_to_co_extinction",
    "survival_summary",
    "phase_share",
    "intervention_counts",
    "intervention_timing",
    "platform_update_summary",
)


def read_parquet_if_exists(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if path.exists():
        try:
            return pd.read_csv(path)
        except EmptyDataError:
            return pd.DataFrame()
    return pd.DataFrame()


def load_stage_outputs(input_dir: str | Path) -> dict[str, pd.DataFrame]:
    root = Path(input_dir)
    return {
        name: read_parquet_if_exists(root / f"{name}.parquet")
        for name in STAGE_TABLES
    }


def load_analysis_outputs(analysis_dir: str | Path) -> dict[str, pd.DataFrame]:
    root = Path(analysis_dir)
    return {
        name: read_csv_if_exists(root / f"{name}.csv")
        for name in ANALYSIS_TABLES
    }
