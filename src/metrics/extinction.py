import numpy as np
from ._utils import clip01, normalize_weights

def recovery_probability(*, valid_reproduction, error_reproduction, lineage_diversity, verification_capacity, external_input_capacity, weights=(.25,.25,.20,.15,.15)):
    w=normalize_weights(weights,5); x=[clip01(valid_reproduction/(1+valid_reproduction)),clip01(1/(1+error_reproduction)),clip01(lineage_diversity/(1+lineage_diversity)),clip01(verification_capacity),clip01(external_input_capacity)]
    return clip01(float(np.dot(w,x)))
def co_extinction_risk(*, valid_vanishing, distortion, repetition, error_persistence_value, performance_decline, recovery_probability_value, weights=(.20,.20,.15,.25,.20)):
    w=normalize_weights(weights,5); loss=float(np.dot(w,[clip01(v) for v in [valid_vanishing,distortion,repetition,error_persistence_value,performance_decline]]))
    return clip01(loss*(1-clip01(recovery_probability_value)))
def extinction_state(*, quality, diversity, error_level, recovery_probability_value, quality_threshold=.30, diversity_threshold=.20, error_threshold=.70, recovery_threshold=.20):
    return bool(quality<quality_threshold and diversity<diversity_threshold and error_level>error_threshold and recovery_probability_value<recovery_threshold)
