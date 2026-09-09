from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.environment import RecursiveModelConfig, run_recursive_interaction
from src.scenarios import ScenarioRuntime, load_scenario


def main() -> None:
    scenario = load_scenario(Path("config/scenarios/S2_critic_feedback.yaml"))
    config = RecursiveModelConfig(
        num_steps=25,
        seed=42,
        run_id="quick_demo",
        scenario_id=scenario.id,
        ai_feedback_reuse=scenario.interaction.ai_reuse_ratio,
    )

    result = run_recursive_interaction(
        config,
        scenario_runtime=ScenarioRuntime(scenario),
    )
    frames = result.as_frames()
    interactions = frames["events"][
        frames["events"]["event_type"].eq("interaction_cycle")
    ]
    interventions = frames["events"][
        frames["events"]["is_intervention"].fillna(False)
    ]

    print("Final interaction states")
    print(
        interactions[
            [
                "time",
                "human_input_validity",
                "ai_output_validity",
                "final_validity",
                "recursive_ai_input_share",
                "source_independence",
                "co_thinking",
                "co_movement",
            ]
        ]
        .tail(8)
        .to_string(index=False)
    )
    print(f"\nIntervention events: {len(interventions)}")


if __name__ == "__main__":
    main()
