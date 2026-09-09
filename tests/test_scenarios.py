from pathlib import Path
from src.scenarios import load_scenarios, load_scenario, ScenarioRuntime

D = Path("config/scenarios")

def test_all_load():
    xs = load_scenarios(D)
    assert len(xs) == 8

def test_human_injection_trigger():
    s = load_scenario(D/"S1_human_data_injection.yaml")
    r = ScenarioRuntime(s)
    assert not r.decide(24, {}).triggered
    assert r.decide(25, {}).interaction_action == "human_data_injection"

def test_critic_threshold():
    s = load_scenario(D/"S2_critic_feedback.yaml")
    r = ScenarioRuntime(s)
    assert not r.decide(15, {"repetition_rate":.3}).triggered
    assert r.decide(16, {"repetition_rate":.6}).interaction_action == "critic_feedback"

def test_adaptive_inbreeding():
    s = load_scenario(D/"S6_adaptive_intervention.yaml")
    r = ScenarioRuntime(s)
    d = r.decide(25,{
        "recursive_degradation_risk":.7,
        "lineage_concentration":.8,
        "diversity_loss":.6,
        "context_contamination":.1,
        "model_contamination":.1,
        "R_E":.8,
        "recovery_probability":.5,
    })
    assert d.diagnosis == "epistemic_inbreeding"
    assert d.interaction_action == "human_data_injection"
