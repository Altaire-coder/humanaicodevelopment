from dataclasses import dataclass, field
from typing import Literal

TriggerType = Literal["none","fixed_time","periodic","threshold","predictive"]

@dataclass(frozen=True)
class TriggerPolicy:
    type: TriggerType = "none"
    start_time: int = 0
    interval: int | None = None
    threshold_metric: str | None = None
    threshold_value: float | None = None
    prediction_horizon: int | None = None
    failure_probability_threshold: float | None = None
    cooldown: int = 0

    def validate(self):
        if self.start_time < 0 or self.cooldown < 0:
            raise ValueError("Times must be non-negative.")
        if self.type == "periodic" and (self.interval is None or self.interval <= 0):
            raise ValueError("Periodic trigger requires interval > 0.")
        if self.type == "threshold" and (
            self.threshold_metric is None or self.threshold_value is None
        ):
            raise ValueError("Threshold trigger requires metric and value.")
        if self.type == "predictive" and (
            self.prediction_horizon is None
            or self.failure_probability_threshold is None
        ):
            raise ValueError("Predictive trigger requires horizon and probability.")

@dataclass(frozen=True)
class InteractionPolicy:
    action: str = "none"
    trigger: TriggerPolicy = field(default_factory=TriggerPolicy)
    ai_reuse_ratio: float = 0.75
    human_data_injection: float = 0.0
    injected_data_validity: float = 0.9
    injected_data_novelty: float = 0.8
    injected_data_alignment: float = 0.95
    source_independence: float = 1.0
    critic_strength: float = 0.0
    counterfactual_strength: float = 0.0
    uncertainty_flagging: float = 0.0
    intention_anchor_strength: float = 0.0
    context_reset_strength: float = 0.0
    preserve_verified_summary: bool = True

    def validate(self):
        self.trigger.validate()
        for name, value in self.__dict__.items():
            if isinstance(value, float) and name != "ai_reuse_ratio":
                if not 0 <= value <= 1:
                    raise ValueError(f"{name} must be in [0,1].")

@dataclass(frozen=True)
class PlatformPolicy:
    action: str = "none"
    trigger: TriggerPolicy = field(default_factory=TriggerPolicy)
    learning_enabled: bool = False
    update_interval: int = 10
    interaction_sampling_rate: float = 0.3
    learning_rate: float = 0.1
    human_training_share: float = 0.35
    ai_training_share: float = 0.15
    verified_training_share: float = 0.5
    contaminated_training_share: float = 0.0
    rollback_steps: int = 0
    reset_strength: float = 0.0
    preserve_verified_knowledge: bool = True

    def validate(self):
        self.trigger.validate()
        shares = [
            self.human_training_share,
            self.ai_training_share,
            self.verified_training_share,
            self.contaminated_training_share,
        ]
        if any(not 0 <= x <= 1 for x in shares):
            raise ValueError("Training shares must be in [0,1].")
        if sum(shares) > 1.000001:
            raise ValueError("Training shares cannot exceed 1.")
        if self.update_interval <= 0:
            raise ValueError("update_interval must be positive.")

@dataclass(frozen=True)
class ScenarioConfig:
    id: str
    name: str
    description: str
    expected_mechanism: str
    interaction: InteractionPolicy = field(default_factory=InteractionPolicy)
    platform: PlatformPolicy = field(default_factory=PlatformPolicy)
    diagnosis_enabled: bool = False
    adaptive_actions: dict[str, str] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)

    def validate(self):
        self.interaction.validate()
        self.platform.validate()
        if self.diagnosis_enabled and not self.adaptive_actions:
            raise ValueError("Adaptive scenario requires adaptive_actions.")
