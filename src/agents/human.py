from dataclasses import dataclass

@dataclass
class HumanAgent:
    agent_id: int
    knowledge: float
    accuracy: float
    novelty: float
    verification: float
    confirmation_bias: float
    ai_reliance: float
    confidence_calibration: float
    independent_performance: float

    def generate_input(self, rng, prior_ai_influence: float) -> dict:
        """Generate a probabilistic human idea state."""
        validity = max(0.0, min(1.0, rng.normal(self.accuracy, 0.10)))
        novelty = max(0.0, min(1.0, rng.normal(self.novelty, 0.10)))
        return {
            "validity": validity,
            "novelty": novelty,
            "ai_origin_share": max(0.0, min(1.0, prior_ai_influence)),
        }

    def evaluate_ai_output(self, output: dict, rng) -> str:
        """Return one of accept, verify, correct, challenge, reject, explore."""
        if output["validity"] < 0.5 and rng.random() < self.verification:
            return "correct"
        if rng.random() < self.confirmation_bias * output.get("agreement", 0.5):
            return "accept"
        return rng.choice(["accept", "verify", "challenge", "explore"])
