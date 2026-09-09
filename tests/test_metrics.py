import numpy as np, pandas as pd
from src.metrics import *

def test_core():
    assert valid_novelty(.9,.9,.9)>valid_novelty(.9,.2,.9)
    a=co_thinking(complementarity=.8,valid_novelty_score=.7,revision_depth=.6,perspective_expansion=.5,copying=.1)
    b=co_thinking(complementarity=.8,valid_novelty_score=.7,revision_depth=.6,perspective_expansion=.5,copying=.9)
    assert a>b and quality_adjusted_comovement(comovement=.9,delta_joint_quality=-.4)<0

def test_diversity_accuracy():
    assert lineage_concentration([.5,.5])==.5 and effective_lineage_diversity([.5,.5])==2
    assert repetition_rate([.95,.91,.4],.9)==2/3
    assert np.isclose(calibration_error([.9,.2],[1,0]),.15)

def test_genealogy_convergence():
    df=pd.DataFrame([{'idea_id':'I0','parent_idea_id':None,'validity':.9,'novelty':.8,'error_severity':.1},{'idea_id':'I1','parent_idea_id':'I0','validity':.8,'novelty':.6,'error_severity':.2},{'idea_id':'I2','parent_idea_id':'I0','validity':.3,'novelty':.5,'error_severity':.8}])
    r=reproduction_numbers(df); assert set(r)=={'R_V','R_N','R_E'}
    base=np.array([[1,0],[0,1]]); cur=np.array([[1,0],[.9,.1]])
    assert semantic_convergence(cur,base)>0

def test_development_extinction_risk():
    p=preservation_score(valid_survival=.9,intention_survival=.8,independent_human_performance=.7)
    e=productive_expansion_score(valid_diversity=.8,valid_creativity=.7,utility_gain=.6)
    assert 0<co_development_score(p,e)<=1
    assert 0<co_degradation_score(error_growth=.8,diversity_loss=.7,dependence_growth=.6,intention_drift=.5,independent_capability_decline=.4)<=1
    assert net_epistemic_reproduction(1.2,.7)==.5
    rp=recovery_probability(valid_reproduction=1.2,error_reproduction=.5,lineage_diversity=3,verification_capacity=.8,external_input_capacity=.7)
    ce=co_extinction_risk(valid_vanishing=.8,distortion=.7,repetition=.6,error_persistence_value=.8,performance_decline=.7,recovery_probability_value=rp)
    assert 0<=rp<=1 and 0<=ce<=1

def test_recursive_risk():
    r=recursive_degradation_risk(lineage_concentration=.8,diversity_loss=.7,error_persistence_value=.6,error_reproduction=1.3,intention_drift=.5,performance_decline=.6,human_reliance=.8,recovery_probability_value=.2,tail_survival=.3)
    u=intervention_urgency(current_risk=r,previous_risk=max(0,r-.1),risk_two_steps_back=max(0,r-.15))
    assert 0<=r<=1 and 0<=u<=1
