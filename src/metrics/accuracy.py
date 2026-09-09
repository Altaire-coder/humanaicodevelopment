import numpy as np

def calibration_error(confidence,outcomes):
    c=np.asarray(confidence,dtype=float); o=np.asarray(outcomes,dtype=float)
    if len(c)!=len(o): raise ValueError("length mismatch")
    return 0.0 if len(c)==0 else float(np.mean(np.abs(c-o)))
def brier_score(probabilities,outcomes):
    p=np.asarray(probabilities,dtype=float); o=np.asarray(outcomes,dtype=float)
    if len(p)!=len(o): raise ValueError("length mismatch")
    return 0.0 if len(p)==0 else float(np.mean((p-o)**2))
def success_rate(values,threshold=.5):
    x=np.asarray(values,dtype=float); return 0.0 if len(x)==0 else float(np.mean(x>=threshold))
def error_persistence(error_flags_by_time):
    e=np.asarray(error_flags_by_time,dtype=bool)
    if e.ndim!=2 or e.shape[0]<2:return 0.0
    d=e[:-1].sum(); return 0.0 if d==0 else float(np.logical_and(e[:-1],e[1:]).sum()/d)
