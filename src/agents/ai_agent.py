from dataclasses import dataclass

@dataclass
class AIAgent:
    agent_id: int
    accuracy: float
    generative_breadth: float
    sycophancy: float
    memory_strength: float
    personalization: float
    copying_fidelity: float
    confidence_calibration: float

    def respond(self, human_input: dict, rng, context_contamination: float = 0.0) -> dict:
        base = self.accuracy * (1.0 - context_contamination)
        validity = max(0.0, min(1.0, rng.normal(base, 0.10)))
        novelty = max(0.0, min(1.0, rng.normal(self.generative_breadth, 0.10)))
        agreement = max(0.0, min(1.0, self.sycophancy + 0.5 * human_input["validity"]))
        confidence = max(
            0.0,
            min(1.0, validity * self.confidence_calibration + (1-self.confidence_calibration))
        )
        return {
            "validity": validity,
            "novelty": novelty,
            "agreement": agreement,
            "confidence": confidence,
        }
