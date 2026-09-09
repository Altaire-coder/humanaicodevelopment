import numpy as np

def reproduction_numbers(ideas, validity_threshold=.60, novelty_threshold=.70, error_threshold=.40):
    req={'idea_id','parent_idea_id','validity','novelty','error_severity'}
    miss=req-set(ideas.columns)
    if miss: raise ValueError(f"Missing columns: {sorted(miss)}")
    counts=ideas['parent_idea_id'].value_counts().to_dict(); rv=[]; rn=[]; re=[]
    for _,r in ideas.iterrows():
        n=int(counts.get(r['idea_id'],0))
        if r['validity']>=validity_threshold and r['error_severity']<error_threshold: rv.append(n)
        if r['novelty']>=novelty_threshold: rn.append(n)
        if r['validity']<validity_threshold or r['error_severity']>=error_threshold: re.append(n)
    return {'R_V':float(np.mean(rv)) if rv else 0.0,'R_N':float(np.mean(rn)) if rn else 0.0,'R_E':float(np.mean(re)) if re else 0.0}

def valid_idea_survival_rate(idea_states, previous_time, current_time, validity_threshold=.60):
    p=set(idea_states[(idea_states.time==previous_time)&(idea_states.validity>=validity_threshold)&(idea_states.alive)].idea_id)
    c=set(idea_states[(idea_states.time==current_time)&(idea_states.alive)].idea_id)
    return 1.0 if not p else float(len(p&c)/len(p))
def idea_vanishing_rate(idea_states, previous_time, current_time):
    p=set(idea_states[(idea_states.time==previous_time)&(idea_states.alive)].idea_id)
    c=set(idea_states[(idea_states.time==current_time)&(idea_states.alive)].idea_id)
    return 0.0 if not p else float(len(p-c)/len(p))
