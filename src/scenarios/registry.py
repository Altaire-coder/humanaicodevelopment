SCENARIOS = {
    "S0_no_treatment": {
        "human_data_injection": 0.0,
        "critic_feedback": False,
        "context_reset": False,
        "platform_retraining": False,
        "rollback": False,
    },
    "S1_human_data_injection": {
        "human_data_injection": 0.2,
        "critic_feedback": False,
        "context_reset": False,
        "platform_retraining": False,
        "rollback": False,
    },
    "S2_critic_feedback": {
        "human_data_injection": 0.0,
        "critic_feedback": True,
        "context_reset": False,
        "platform_retraining": False,
        "rollback": False,
    },
    "S3_context_reset": {
        "human_data_injection": 0.0,
        "critic_feedback": False,
        "context_reset": True,
        "platform_retraining": False,
        "rollback": False,
    },
    "S4_verified_retraining": {
        "human_data_injection": 0.0,
        "critic_feedback": False,
        "context_reset": False,
        "platform_retraining": True,
        "rollback": False,
    },
    "S6_model_rollback": {
        "human_data_injection": 0.0,
        "critic_feedback": False,
        "context_reset": False,
        "platform_retraining": False,
        "rollback": True,
    },
}
