from dataclasses import dataclass, field

@dataclass(frozen=True)
class ScenarioDecision:
    time: int
    triggered: bool
    diagnosis: str
    interaction_action: str
    platform_action: str
    treatment_level: str | None
    treatment_id: str | None
    reason: str
    parameters: dict = field(default_factory=dict)

class ScenarioRuntime:
    def __init__(self, scenario, max_interventions: int | None = None):
        self.scenario = scenario
        self.last_intervention_time = None
        self.max_interventions = max_interventions
        self.intervention_count = 0

    def _triggered(self, trigger, time, metrics):
        if trigger.type == "none":
            return False, "No trigger."
        if time < trigger.start_time:
            return False, "Before start."
        if self.last_intervention_time is not None and (
            time - self.last_intervention_time < trigger.cooldown
        ):
            return False, "Cooldown."
        if trigger.type == "fixed_time":
            return time == trigger.start_time, "Fixed time."
        if trigger.type == "periodic":
            return (time - trigger.start_time) % trigger.interval == 0, "Periodic."
        if trigger.type == "threshold":
            value = metrics.get(trigger.threshold_metric)
            return (
                value is not None and value >= trigger.threshold_value,
                f"{trigger.threshold_metric}={value}",
            )
        if trigger.type == "predictive":
            value = metrics.get("predicted_failure_probability")
            return (
                value is not None and value >= trigger.failure_probability_threshold,
                f"predicted_failure_probability={value}",
            )
        return False, "Unsupported."

    @staticmethod
    def diagnose(m):
        if m.get("recursive_degradation_risk", 0) >= .85 and m.get("R_E", 0) > 1 and m.get("recovery_probability", 1) < .2:
            return "critical_failure"
        if m.get("model_contamination", 0) >= .5 or m.get("cross_session_error_rate", 0) >= .4:
            return "platform_contamination"
        if m.get("lineage_concentration", 0) >= .65 and m.get("diversity_loss", 0) >= .4:
            return "epistemic_inbreeding"
        if m.get("context_contamination", 0) >= .5 and m.get("model_contamination", 0) < .3:
            return "local_context_contamination"
        if m.get("utility_quality_gap", 0) >= .35:
            return "structural_utility_failure"
        return "stable"

    def decide(self, time, metrics):
        diagnosis = self.diagnose(metrics) if self.scenario.diagnosis_enabled else "not_applicable"
        it, ir = self._triggered(self.scenario.interaction.trigger, time, metrics)
        pt, pr = self._triggered(self.scenario.platform.trigger, time, metrics)
        ia = self.scenario.interaction.action if it else "none"
        pa = self.scenario.platform.action if pt else "none"

        if self.scenario.diagnosis_enabled:
            action = self.scenario.adaptive_actions.get(diagnosis, "none")
            if action in {"human_data_injection","critic_feedback","context_reset"}:
                ia, it = action, action != "none"
            elif action in {"verified_retraining","model_rollback","model_reset"}:
                pa, pt = action, action != "none"

        triggered = it or pt
        if (
            triggered
            and self.max_interventions is not None
            and self.intervention_count >= self.max_interventions
        ):
            triggered = False
            ia = "none"
            pa = "none"
            it = False
            pt = False
            ir = "Intervention cap reached."
            pr = "Intervention cap reached."
        if triggered:
            self.last_intervention_time = time
            self.intervention_count += 1
        level = "interaction_and_platform" if it and pt else "interaction" if it else "platform" if pt else None
        return ScenarioDecision(
            time=time, triggered=triggered, diagnosis=diagnosis,
            interaction_action=ia, platform_action=pa,
            treatment_level=level,
            treatment_id=f"{self.scenario.id}_T{time:04d}" if triggered else None,
            reason=f"{ir}; {pr}",
            parameters={
                "human_data_injection": self.scenario.interaction.human_data_injection,
                "critic_strength": self.scenario.interaction.critic_strength,
                "context_reset_strength": self.scenario.interaction.context_reset_strength,
                "verified_training_share": self.scenario.platform.verified_training_share,
                "rollback_steps": self.scenario.platform.rollback_steps,
            },
        )
