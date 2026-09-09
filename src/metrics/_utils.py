import numpy as np

def clip01(v): return float(np.clip(v,0.0,1.0))
def normalize_weights(weights,n):
    w=np.asarray(weights,dtype=float)
    if len(w)!=n or np.any(w<0) or w.sum()<=0: raise ValueError("Invalid weights")
    return w/w.sum()
