import numpy as np

def _d(a,b):
    z=np.linalg.norm(a)*np.linalg.norm(b)
    return 0.0 if z==0 else float(1-np.dot(a,b)/z)
def pairwise_divergence(embeddings):
    x=np.asarray(embeddings,dtype=float)
    if len(x)<2:return 0.0
    return float(np.mean([_d(x[i],x[j]) for i in range(len(x)) for j in range(i+1,len(x))]))
def semantic_convergence(current_embeddings, baseline_embeddings):
    c=pairwise_divergence(current_embeddings); b=pairwise_divergence(baseline_embeddings)
    return 0.0 if b==0 else float(1-c/b)
def within_group_convergence(embeddings, group_labels):
    x=np.asarray(embeddings,dtype=float); g=np.asarray(group_labels); vals=[]; weights=[]
    for k in np.unique(g):
        s=x[g==k]
        if len(s)>=2: vals.append(pairwise_divergence(s)); weights.append(len(s))
    return 0.0 if not vals else float(1-np.average(vals,weights=weights))
def between_group_divergence(embeddings, group_labels):
    x=np.asarray(embeddings,dtype=float); g=np.asarray(group_labels)
    cent=np.asarray([x[g==k].mean(axis=0) for k in np.unique(g)])
    return pairwise_divergence(cent)
