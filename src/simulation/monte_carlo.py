from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import pandas as pd

from src.environment import RecursiveModelConfig, run_recursive_interaction


@dataclass
class MonteCarloConfig:
    runs: int = 100
    base_seed: int = 42
    output_dir: str = "outputs/monte_carlo"
    save_individual_tables: bool = True


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


def _combine_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    nonempty = [frame for frame in frames if not frame.empty]
    if nonempty:
        return pd.concat(nonempty, ignore_index=True)
    return pd.DataFrame()


def _empty_platform_update_events() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "run_id",
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
        ]
    )


def run_monte_carlo(
    model_config: RecursiveModelConfig,
    mc_config: MonteCarloConfig,
) -> dict[str, pd.DataFrame]:
    output_dir = Path(mc_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    collected: dict[str, list[pd.DataFrame]] = {
        name: [] for name in OUTPUT_TABLES
    }

    for run_index in range(mc_config.runs):
        cfg = replace(
            model_config,
            seed=mc_config.base_seed + run_index,
            run_id=f"R_{run_index + 1:05d}",
        )
        result = run_recursive_interaction(cfg)
        frames = result.as_frames()

        missing = set(OUTPUT_TABLES) - set(frames)
        if missing:
            raise KeyError(
                "RecursiveSimulationResult.as_frames() is missing "
                f"Stage 7 tables: {sorted(missing)}"
            )

        for name in OUTPUT_TABLES:
            collected[name].append(frames[name])

        if mc_config.save_individual_tables:
            run_dir = output_dir / f"run_{run_index + 1:05d}"
            run_dir.mkdir(parents=True, exist_ok=True)
            for name in OUTPUT_TABLES:
                frame = frames[name]
                if name == "platform_update_events" and frame.empty:
                    frame = _empty_platform_update_events()
                frame.to_parquet(run_dir / f"{name}.parquet", index=False)

    outputs: dict[str, pd.DataFrame] = {}
    for name in OUTPUT_TABLES:
        combined = _combine_frames(collected[name])

        if name == "platform_update_events" and combined.empty:
            combined = _empty_platform_update_events()

        outputs[name] = combined

        if mc_config.save_individual_tables:
            combined.to_parquet(output_dir / f"{name}.parquet", index=False)

    return outputs
