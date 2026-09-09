from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

from src.environment import RecursiveModelConfig, run_recursive_interaction
from src.scenarios import ScenarioRuntime, load_scenario


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


def load_experiment(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _combine_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    nonempty = [frame for frame in frames if not frame.empty]
    if nonempty:
        return pd.concat(nonempty, ignore_index=True)
    return pd.DataFrame()


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


def _annotate_run_identity(
    frame: pd.DataFrame,
    *,
    matched_run_id: str,
    scenario_index: int,
    run_index: int,
    seed: int,
) -> pd.DataFrame:
    if frame.empty:
        return frame

    out = frame.copy()
    out["matched_run_id"] = matched_run_id
    out["scenario_index"] = scenario_index
    out["run_index"] = run_index
    out["seed"] = seed
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/scenario_experiment.yaml")
    parser.add_argument(
        "--run-start",
        type=int,
        required=True,
        help="One-based global run index to start from, e.g. 1, 21, 41.",
    )
    parser.add_argument(
        "--run-count",
        type=int,
        required=True,
        help="Number of matched runs to execute in this batch.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Batch output directory, e.g. outputs/stage6_batches/batch_001_020.",
    )
    args = parser.parse_args()

    if args.run_start < 1:
        raise ValueError("--run-start must be one-based and >= 1")
    if args.run_count < 1:
        raise ValueError("--run-count must be >= 1")

    raw = load_experiment(args.config)["experiment"]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    configured_runs = int(raw["monte_carlo"]["runs_per_scenario"])
    steps = int(raw["monte_carlo"]["steps"])
    base_seed = int(raw["monte_carlo"]["base_seed"])
    matched_seeds = bool(raw["monte_carlo"].get("matched_seeds", True))

    start_index = args.run_start - 1
    end_index = start_index + args.run_count

    if end_index > configured_runs:
        raise ValueError(
            f"Requested runs {args.run_start}-{end_index}, but config has "
            f"runs_per_scenario={configured_runs}."
        )

    collected: dict[str, list[pd.DataFrame]] = {
        name: [] for name in OUTPUT_TABLES
    }

    total_jobs = len(raw["scenarios"]) * (end_index - start_index)
    progress = tqdm(
        total=total_jobs,
        desc=f"scenario batch {args.run_start}-{end_index}",
        unit="run",
    ) if tqdm is not None else None

    for scenario_index, path in enumerate(raw["scenarios"]):
        scenario = load_scenario(path)

        for global_run_index in range(start_index, end_index):
            matched_run_id = f"R{global_run_index + 1:05d}"

            if progress is not None:
                progress.set_postfix(
                    scenario=scenario.id,
                    run=matched_run_id,
                    refresh=False,
                )
            else:
                print(
                    f"[batch {args.run_start}-{end_index}] "
                    f"{scenario.id} {matched_run_id}"
                )

            if matched_seeds:
                seed = base_seed + global_run_index
            else:
                seed = base_seed + scenario_index * 100000 + global_run_index

            config = RecursiveModelConfig(
                num_steps=steps,
                seed=seed,
                run_id=f"{scenario.id}_{matched_run_id}",
                scenario_id=scenario.id,
                ai_feedback_reuse=scenario.interaction.ai_reuse_ratio,
                platform_learning_enabled=scenario.platform.learning_enabled,
                platform_update_interval=scenario.platform.update_interval,
                platform_interaction_sampling_rate=(
                    scenario.platform.interaction_sampling_rate
                ),
                platform_learning_rate=scenario.platform.learning_rate,
            )

            result = run_recursive_interaction(
                config,
                scenario_runtime=ScenarioRuntime(scenario),
            )
            frames = result.as_frames()

            missing = set(OUTPUT_TABLES) - set(frames)
            if missing:
                raise KeyError(
                    "RecursiveSimulationResult.as_frames() is missing "
                    f"Stage 7 tables: {sorted(missing)}"
                )

            for name in OUTPUT_TABLES:
                collected[name].append(
                    _annotate_run_identity(
                        frames[name],
                        matched_run_id=matched_run_id,
                        scenario_index=scenario_index,
                        run_index=global_run_index,
                        seed=seed,
                    )
                )

            if progress is not None:
                progress.update(1)

    if progress is not None:
        progress.close()

    for name in OUTPUT_TABLES:
        combined = _combine_frames(collected[name])
        if name == "platform_update_events" and combined.empty:
            combined = _empty_platform_update_events()

        out_path = output_dir / f"{name}.parquet"
        combined.to_parquet(out_path, index=False)
        print(f"{name}: {len(combined):,} rows")

    print(
        f"Saved batch runs {args.run_start}-{end_index} "
        f"to {output_dir.resolve()}"
    )
    print(f"matched_seeds: {matched_seeds}")


if __name__ == "__main__":
    main()
