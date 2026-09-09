import numpy as np
from ._utils import clip01, normalize_weights

def _gmean(vals,w):
    x=np.clip(np.asarray(vals,dtype=float),1e-12,1.0); return clip01(float(np.exp(np.sum(w*np.log(x)))))
def preservation_score(*, valid_survival, intention_survival, independent_human_performance, weights=(1,1,1)):
    return _gmean([valid_survival,intention_survival,independent_human_performance],normalize_weights(weights,3))
def productive_expansion_score(*, valid_diversity, valid_creativity, utility_gain, weights=(1,1,1)):
    return _gmean([valid_diversity,valid_creativity,clip01(utility_gain)],normalize_weights(weights,3))
def co_development_score(preservation, productive_expansion): return clip01((max(0,preservation)*max(0,productive_expansion))**0.5)
def co_degradation_score(*, error_growth, diversity_loss, dependence_growth, intention_drift, independent_capability_decline, weights=(.25,.20,.20,.20,.15)):
    w=normalize_weights(weights,5); x=[clip01(v) for v in [error_growth,diversity_loss,dependence_growth,intention_drift,independent_capability_decline]]
    return clip01(float(np.dot(w,x)))
def net_epistemic_reproduction(r_valid,r_error): return float(r_valid-r_error)
