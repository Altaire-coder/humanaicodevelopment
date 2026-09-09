from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_TABLES = (
    "events",
    "idea_states",
    "human_states",
    "ai_states",
    "local_ai_states",
    "platform_states",
    "lineage_edges",
)

OPTIONAL_TABLES = (
    "platform_update_events",
    "outcome_metrics",
)


def read_stage_outputs(input_dir: str | Path) -> dict[str, pd.DataFrame]:
    input_dir = Path(input_dir)
    frames: dict[str, pd.DataFrame] = {}

    missing = []
    for name in REQUIRED_TABLES:
        path = input_dir / f"{name}.parquet"
        if not path.exists():
            missing.append(str(path))
        else:
            frames[name] = pd.read_parquet(path)

    if missing:
        raise FileNotFoundError(
            "Missing required Stage 7/8 output tables:\n"
            + "\n".join(missing)
        )

    for name in OPTIONAL_TABLES:
        path = input_dir / f"{name}.parquet"
        frames[name] = (
            pd.read_parquet(path)
            if path.exists()
            else pd.DataFrame()
        )

    return frames


def ensure_outcome_metrics(
    *,
    frames: dict[str, pd.DataFrame],
    input_dir: str | Path,
    output_path: str | Path | None = None,
):
    if not frames.get("outcome_metrics", pd.DataFrame()).empty:
        return frames["outcome_metrics"]

    from src.metrics.pipeline import compute_run_metrics

    metrics = compute_run_metrics(
        events=frames["events"],
        idea_states=frames["idea_states"],
        human_states=frames["human_states"],
        ai_states=frames["ai_states"],
        platform_states=frames["platform_states"],
    )

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        metrics.to_parquet(output_path, index=False)

    return metrics


def write_table(frame: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix.lower() == ".csv":
        frame.to_csv(path, index=False)
    elif path.suffix.lower() in {".parquet", ".pq"}:
        frame.to_parquet(path, index=False)
    else:
        raise ValueError(f"Unsupported output format: {path.suffix}")
