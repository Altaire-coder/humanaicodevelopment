from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


OUTPUT_TABLES = (
    "events",
    "idea_states",
    "human_states",
    "ai_states",
    "local_ai_states",
    "platform_states",
    "platform_update_events",
    "lineage_edges",
)


def _empty_platform_update_events() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "run_id",
            "matched_run_id",
            "scenario_id",
            "time",
            "event_type",
            "update_reason",
            "sample_size",
            "eligible_event_count",
            "sample_validity",
            "sample_novelty",
            "sample_error",
            "sample_ai_origin_share",
            "model_version_before",
            "model_version_after",
            "scenario_index",
            "run_index",
            "seed",
        ]
    )


def _read_table_from_batches(input_root: Path, table: str) -> pd.DataFrame:
    paths = sorted(input_root.glob(f"*/{table}.parquet"))
    frames: list[pd.DataFrame] = []

    for path in paths:
        frame = pd.read_parquet(path)
        if not frame.empty:
            frame = frame.copy()
            frame["batch_dir"] = path.parent.name
            frames.append(frame)

    if frames:
        combined = pd.concat(frames, ignore_index=True)
        combined = combined.drop_duplicates()
        return combined

    if table == "platform_update_events":
        return _empty_platform_update_events()

    return pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root",
        required=True,
        help="Root containing batch folders, e.g. outputs/stage6_batches.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Combined output directory, e.g. outputs/stage6.",
    )
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_root.exists():
        raise FileNotFoundError(input_root)

    for table in OUTPUT_TABLES:
        combined = _read_table_from_batches(input_root, table)
        out_path = output_dir / f"{table}.parquet"
        combined.to_parquet(out_path, index=False)
        print(f"{table}: {len(combined):,} rows")

    print(f"Combined scenario outputs saved to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
