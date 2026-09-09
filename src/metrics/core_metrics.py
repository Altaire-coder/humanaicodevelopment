import numpy as np
from ._utils import clip01, normalize_weights

def valid_novelty(novelty, validity, alignment=1.0, utility=None):
    vals=[novelty,validity,alignment]+([] if utility is None else [utility])
    return clip01(float(np.prod(vals)))

def co_thinking(*, complementarity, valid_novelty_score, revision_depth, perspective_expansion, copying, weights=(.25,.25,.20,.20,.10)):
    w=normalize_weights(weights,5)
    return clip01(w[0]*complementarity+w[1]*valid_novelty_score+w[2]*revision_depth+w[3]*perspective_expansion-w[4]*copying)

def quality_adjusted_comovement(*, comovement, delta_joint_quality, preserve_sign=True):
    if not -1<=comovement<=1: raise ValueError("comovement must be in [-1,1]")
    s=float(comovement*delta_joint_quality)
    return s if preserve_sign else clip01(s)
