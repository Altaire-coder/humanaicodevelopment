"""Recursive interaction and platform environment."""

from .model import RecursiveModelConfig, RecursiveSimulationResult
from .interaction_loop import run_recursive_interaction
from .platform_loop import update_platform_state

__all__ = [
    "RecursiveModelConfig",
    "RecursiveSimulationResult",
    "run_recursive_interaction",
    "update_platform_state",
]
