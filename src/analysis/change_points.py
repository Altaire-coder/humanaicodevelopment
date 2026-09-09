from __future__ import annotations

import numpy as np
import pandas as pd


def simple_change_points(
    metrics: pd.DataFrame,
    *,
    metric: str,
    min_time: int = 5,
    z_threshold: float = 2.0,
) -> pd.DataFrame:
    rows = []
    for (scenario_id, run_id), group in metrics.groupby(["scenario_id", "run_id"]):
        group = group.sort_values("time")
        values = group[metric].to_numpy(dtype=float)
        times = group["time"].to_numpy()
        if len(values) < min_time + 2:
            continue

        diffs = np.diff(values)
        scale = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0
        if scale == 0.0:
            continue

        for idx, diff in enumerate(diffs, start=1):
            if times[idx] < min_time:
                continue
            z = float(abs(diff) / scale)
            if z >= z_threshold:
                rows.append(
                    {
                        "scenario_id": scenario_id,
                        "run_id": run_id,
                        "metric": metric,
                        "time": int(times[idx]),
                        "delta": float(diff),
                        "z": z,
                    }
                )
    return pd.DataFrame(rows)
