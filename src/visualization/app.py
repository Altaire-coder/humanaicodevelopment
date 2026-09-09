from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

try:
    import streamlit as st
except ImportError:  # pragma: no cover
    st = None

from src.visualization.prototype_utils import (
    SCENARIO_OPTIONS,
    current_regime,
    export_frames_to_directory,
    intervention_recommendation,
    run_prototype_simulation,
    summarize_latest,
)


def _require_streamlit() -> None:
    if st is None:
        raise RuntimeError(
            "Streamlit is not installed. Install it with: python -m pip install streamlit"
        )


def _controls_sidebar() -> tuple[str, int, int, dict[str, Any]]:
    st.sidebar.header("Prototype controls")

    scenario_label = st.sidebar.selectbox(
        "Scenario",
        list(SCENARIO_OPTIONS.keys()),
        index=0,
    )
    steps = st.sidebar.slider("Simulation steps", 20, 200, 100, step=10)
    seed = st.sidebar.number_input("Seed", min_value=0, value=5000, step=1)

    st.sidebar.subheader("Human and AI parameters")
    human_accuracy = st.sidebar.slider("Human accuracy", 0.0, 1.0, 0.70, 0.01)
    ai_accuracy = st.sidebar.slider("AI accuracy", 0.0, 1.0, 0.75, 0.01)
    verification = st.sidebar.slider("Verification", 0.0, 1.0, 0.50, 0.01)
    sycophancy = st.sidebar.slider("Sycophancy", 0.0, 1.0, 0.20, 0.01)
    confidence_calibration = st.sidebar.slider(
        "Confidence calibration", 0.0, 1.0, 0.70, 0.01
    )

    st.sidebar.subheader("Recursive dynamics")
    ai_reuse_ratio = st.sidebar.slider("AI reuse ratio", 0.0, 1.0, 0.75, 0.01)
    mutation_rate = st.sidebar.slider("Mutation rate", 0.0, 1.0, 0.10, 0.01)

    st.sidebar.subheader("Intervention parameters")
    human_data_injection_time = st.sidebar.slider(
        "Human-data injection time", 1, max(steps - 1, 1), min(25, steps - 1)
    )
    intervention_start_time = st.sidebar.slider(
        "Intervention start time", 1, max(steps - 1, 1), min(10, steps - 1)
    )
    intervention_interval = st.sidebar.slider("Intervention interval", 1, 25, 5)
    reset_threshold = st.sidebar.slider("Reset threshold", 0.0, 0.10, 0.02, 0.001)
    reset_cooldown = st.sidebar.slider("Reset cooldown", 1, 50, 15)
    context_reset_strength = st.sidebar.slider(
        "Context reset strength", 0.0, 1.0, 0.50, 0.05
    )
    preserve_verified_summary = st.sidebar.checkbox(
        "Preserve verified summary", value=True
    )

    st.sidebar.subheader("Platform update")
    platform_learning_enabled = st.sidebar.checkbox(
        "Enable platform learning", value=False
    )
    retraining_interval = st.sidebar.slider("Retraining interval", 1, 100, 10)
    platform_learning_rate = st.sidebar.slider(
        "Platform learning rate", 0.0, 0.50, 0.05, 0.01
    )
    platform_interaction_sampling_rate = st.sidebar.slider(
        "Platform interaction sampling rate", 0.0, 1.0, 0.20, 0.01
    )

    controls = {
        "human_accuracy": human_accuracy,
        "ai_accuracy": ai_accuracy,
        "verification": verification,
        "sycophancy": sycophancy,
        "confidence_calibration": confidence_calibration,
        "ai_reuse_ratio": ai_reuse_ratio,
        "mutation_rate": mutation_rate,
        "human_data_injection_time": human_data_injection_time,
        "intervention_start_time": intervention_start_time,
        "intervention_interval": intervention_interval,
        "reset_threshold": reset_threshold,
        "reset_cooldown": reset_cooldown,
        "context_reset_strength": context_reset_strength,
        "preserve_verified_summary": preserve_verified_summary,
        "platform_learning_enabled": platform_learning_enabled,
        "retraining_interval": retraining_interval,
        "platform_learning_rate": platform_learning_rate,
        "platform_interaction_sampling_rate": platform_interaction_sampling_rate,
    }

    return scenario_label, int(steps), int(seed), controls


def _get_frames() -> dict[str, pd.DataFrame]:
    return st.session_state.get("frames", {})


def _run_button(scenario_label: str, steps: int, seed: int, controls: dict[str, Any]) -> None:
    if st.sidebar.button("Run simulation", type="primary"):
        scenario_path = SCENARIO_OPTIONS[scenario_label]
        with st.spinner("Running prototype simulation..."):
            frames = run_prototype_simulation(
                scenario_path=scenario_path,
                steps=steps,
                seed=seed,
                controls=controls,
            )
        st.session_state["frames"] = frames
        st.session_state["scenario_label"] = scenario_label
        st.success("Simulation complete.")


def _metric_line_chart(metrics: pd.DataFrame, metric: str, title: str) -> None:
    if metrics.empty or metric not in metrics.columns:
        st.info(f"{metric} is unavailable.")
        return

    data = metrics[["time", metric]].dropna().set_index("time")
    st.subheader(title)
    st.line_chart(data)


def page_home() -> None:
    st.title("Recursive Human–AI Co-development Prototype")
    st.write(
        "This dashboard is a prototype layer. It calls the same simulation and "
        "metrics functions used by command-line experiments."
    )

    frames = _get_frames()
    if not frames:
        st.info("Use the sidebar to configure and run a simulation.")
        return

    metrics = frames.get("outcome_metrics", pd.DataFrame())
    local_ai = frames.get("local_ai_states", pd.DataFrame())

    st.subheader("Current developmental regime")
    st.metric("Regime", current_regime(metrics))
    st.write(intervention_recommendation(metrics, local_ai))

    summary = summarize_latest(metrics)
    if summary:
        cols = st.columns(4)
        display_keys = [
            ("co_development_score", "Co-development"),
            ("recursive_degradation_risk", "Recursive risk"),
            ("R_E", "Error-bearing parent activity"),
            ("recovery_probability", "Recovery probability"),
        ]
        for col, (key, label) in zip(cols, display_keys):
            if key in summary:
                col.metric(label, f"{summary[key]:.3f}")


def page_scenario_setup() -> None:
    st.title("Scenario Setup")
    st.write("Current sidebar controls are used to build the simulation configuration.")
    scenario_label = st.session_state.get("scenario_label", "No run yet")
    st.write(f"Active scenario: **{scenario_label}**")
    st.json(st.session_state.get("last_controls", {}))


def page_live_simulation() -> None:
    st.title("Live Simulation Indicators")
    frames = _get_frames()
    metrics = frames.get("outcome_metrics", pd.DataFrame())

    if metrics.empty:
        st.info("Run a simulation first.")
        return

    _metric_line_chart(metrics, "co_development_score", "Co-development")
    _metric_line_chart(metrics, "recursive_degradation_risk", "Recursive degradation risk")
    _metric_line_chart(metrics, "co_extinction_risk", "Co-extinction risk")
    _metric_line_chart(metrics, "recovery_probability", "Recovery probability")


def page_idea_genealogy() -> None:
    st.title("Idea Genealogy")
    frames = _get_frames()
    idea_states = frames.get("idea_states", pd.DataFrame())
    lineage_edges = frames.get("lineage_edges", pd.DataFrame())

    if idea_states.empty:
        st.info("Run a simulation first.")
        return

    st.subheader("Idea-state table")
    st.dataframe(idea_states.tail(200), use_container_width=True)

    st.subheader("Idea quality over time")
    cols = [c for c in ["validity", "novelty", "alignment", "error_severity"] if c in idea_states.columns]
    if "time" in idea_states.columns and cols:
        st.line_chart(idea_states[["time"] + cols].set_index("time"))

    if not lineage_edges.empty:
        st.subheader("Lineage edges")
        st.dataframe(lineage_edges.tail(200), use_container_width=True)


def page_convergence_divergence() -> None:
    st.title("Convergence–Divergence Map")
    frames = _get_frames()
    metrics = frames.get("outcome_metrics", pd.DataFrame())

    if metrics.empty:
        st.info("Run a simulation first.")
        return

    required = ["recursive_degradation_risk", "co_development_score"]
    if not all(col in metrics.columns for col in required):
        st.info("Required phase-space metrics are unavailable.")
        return

    st.scatter_chart(
        metrics,
        x="recursive_degradation_risk",
        y="co_development_score",
        color="scenario_id" if "scenario_id" in metrics.columns else None,
    )


def page_intervention_comparison() -> None:
    st.title("Intervention Comparison")
    frames = _get_frames()
    events = frames.get("events", pd.DataFrame())

    if events.empty:
        st.info("Run a simulation first.")
        return

    if "is_intervention" not in events.columns:
        st.info("No intervention column found.")
        return

    interventions = events[events["is_intervention"].fillna(False)].copy()
    if interventions.empty:
        st.info("No interventions were triggered in this run.")
        return

    st.subheader("Triggered interventions")
    st.dataframe(interventions, use_container_width=True)

    if {"event_type", "time"}.issubset(interventions.columns):
        counts = interventions.groupby("event_type").size().rename("count").reset_index()
        st.subheader("Intervention counts")
        st.bar_chart(counts.set_index("event_type"))


def page_export_results() -> None:
    st.title("Export Results")
    frames = _get_frames()
    if not frames:
        st.info("Run a simulation first.")
        return

    output_dir = st.text_input(
        "Export directory",
        value="outputs/prototype_export",
    )

    if st.button("Export current run as CSV"):
        saved = export_frames_to_directory(frames, output_dir)
        st.success(f"Exported {len(saved)} tables.")
        for path in saved:
            st.write(str(path))


def main() -> None:
    _require_streamlit()

    st.set_page_config(
        page_title="Human–AI Co-development Prototype",
        layout="wide",
    )

    scenario_label, steps, seed, controls = _controls_sidebar()
    st.session_state["last_controls"] = controls
    _run_button(scenario_label, steps, seed, controls)

    pages = {
        "Home": page_home,
        "Scenario Setup": page_scenario_setup,
        "Live Simulation": page_live_simulation,
        "Idea Genealogy": page_idea_genealogy,
        "Convergence–Divergence Map": page_convergence_divergence,
        "Intervention Comparison": page_intervention_comparison,
        "Export Results": page_export_results,
    }

    choice = st.sidebar.radio("Page", list(pages.keys()))
    pages[choice]()


if __name__ == "__main__":
    main()
