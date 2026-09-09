"""Single-cycle human–AI interaction engine."""

from .interaction import InteractionConfig, InteractionResult, run_interaction
from .evaluation import EvaluationResult, evaluate_ai_response
from .correction import RevisionResult, revise_idea

__all__ = [
    "InteractionConfig",
    "InteractionResult",
    "EvaluationResult",
    "RevisionResult",
    "evaluate_ai_response",
    "revise_idea",
    "run_interaction",
]
