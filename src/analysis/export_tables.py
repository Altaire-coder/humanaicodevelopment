from __future__ import annotations

from pathlib import Path

import pandas as pd


def export_table_bundle(
    tables: dict[str, pd.DataFrame],
    output_dir: str | Path,
    *,
    csv: bool = True,
    parquet: bool = True,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, frame in tables.items():
        if csv:
            frame.to_csv(output_dir / f"{name}.csv", index=False)
        if parquet:
            frame.to_parquet(output_dir / f"{name}.parquet", index=False)
