import numpy as np
from ._utils import clip01, normalize_weights

def recursive_degradation_risk(*, lineage_concentration, diversity_loss, error_persistence_value, error_reproduction, intention_drift, performance_decline, human_reliance, recovery_probability_value, tail_survival, weights=None):
    x=[clip01(lineage_concentration),clip01(diversity_loss),clip01(error_persistence_value),clip01(error_reproduction/(1+error_reproduction)),clip01(intention_drift),clip01(performance_decline),clip01(human_reliance),clip01(1-recovery_probability_value),clip01(1-tail_survival)]
    w=normalize_weights([1]*9 if weights is None else weights,9); return clip01(float(np.dot(w,x)))
def intervention_urgency(*, current_risk, previous_risk, risk_two_steps_back, trend_weight=.25, acceleration_weight=.10):
    trend=current_risk-previous_risk; acc=trend-(previous_risk-risk_two_steps_back)
    return clip01(current_risk+trend_weight*trend+acceleration_weight*acc)
