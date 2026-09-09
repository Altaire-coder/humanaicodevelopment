import numpy as np

def shannon_entropy(probabilities):
    p=np.asarray(probabilities,dtype=float); p=p[p>0]
    if len(p)==0:return 0.0
    p=p/p.sum(); return float(-(p*np.log(p)).sum())
def effective_diversity(probabilities): return float(np.exp(shannon_entropy(probabilities)))
def lineage_concentration(probabilities):
    p=np.asarray(probabilities,dtype=float)
    if len(p)==0 or p.sum()<=0:return 0.0
    p=p/p.sum(); return float((p**2).sum())
def effective_lineage_diversity(probabilities):
    c=lineage_concentration(probabilities); return 0.0 if c==0 else float(1/c)
def repetition_rate(similarity_scores, threshold=.90):
    x=np.asarray(similarity_scores,dtype=float); return 0.0 if len(x)==0 else float(np.mean(x>=threshold))
def tail_idea_survival(initial_frequencies, alive_mask, quantile=.25):
    f=np.asarray(initial_frequencies,dtype=float); a=np.asarray(alive_mask,dtype=bool)
    if len(f)!=len(a): raise ValueError("length mismatch")
    if len(f)==0:return 0.0
    tail=f<=np.quantile(f,quantile); return 1.0 if tail.sum()==0 else float(a[tail].mean())
