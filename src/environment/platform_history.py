from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass, field
from .model import AIState, PlatformState

@dataclass
class PlatformCheckpoint:
    time: int
    platform: PlatformState
    global_ai: AIState
    reason: str

@dataclass
class PlatformHistory:
    checkpoints: list[PlatformCheckpoint] = field(default_factory=list)

    def save(self, *, time: int, platform: PlatformState, global_ai: AIState, reason: str) -> None:
        self.checkpoints.append(
            PlatformCheckpoint(
                time=time,
                platform=deepcopy(platform),
                global_ai=deepcopy(global_ai),
                reason=reason,
            )
        )

    def rollback(self, *, steps: int) -> tuple[PlatformState, AIState]:
        if steps <= 0:
            raise ValueError("steps must be positive.")
        if not self.checkpoints:
            raise ValueError("No platform checkpoints available.")
        index = max(0, len(self.checkpoints)-1-steps)
        cp = self.checkpoints[index]
        return deepcopy(cp.platform), deepcopy(cp.global_ai)
