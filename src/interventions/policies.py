def select_intervention(diagnosis: str) -> str:
    mapping = {
        "local_context_contamination": "context_reset",
        "epistemic_inbreeding": "human_data_injection",
        "platform_contamination": "verified_retraining",
        "structural_utility_failure": "reward_rule_change",
        "critical_failure": "rollback_or_full_reset",
    }
    return mapping.get(diagnosis, "monitor")

def threshold_policy(urgency: float) -> str:
    if urgency < 0.30:
        return "monitor"
    if urgency < 0.50:
        return "add_diverse_retrieval"
    if urgency < 0.70:
        return "partial_context_reset"
    if urgency < 0.85:
        return "verified_retraining_or_switch"
    return "rollback_and_full_audit"
