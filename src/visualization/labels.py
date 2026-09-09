from __future__ import annotations
SCENARIO_LABELS={"S0_no_treatment":"S0 No treatment","S1_human_data_injection":"S1 Human data","S2_critic_feedback":"S2 Critic feedback","S3_context_reset":"S3 Context reset"}
EVENT_LABELS={"human_data_injection":"Human data injection","critic_feedback":"Critic feedback","context_reset":"Context reset"}
PHASE_LABELS={"low_activity_stable":"Stable low-activity","productive_co_development":"Productive co-development","productive_but_fragile":"Productive but fragile","recursive_degradation":"Recursive degradation"}
METRIC_LABELS={"co_development_score":"Co-development","co_degradation_score":"Co-degradation","co_extinction_risk":"Co-extinction risk","recursive_degradation_risk":"Recursive degradation risk","R_V":"Valid lineage activity","R_N":"Novel lineage activity","R_E":"Error-bearing parent activity","NER":"Net epistemic reproduction","recovery_probability":"Recovery probability","mean_validity":"Validity","mean_novelty":"Novelty","mean_alignment":"Alignment","human_ai_reliance":"Human-AI reliance","model_contamination":"Model contamination"}
MAIN_METRICS=["co_development_score","co_degradation_score","co_extinction_risk","recursive_degradation_risk","R_E","NER","recovery_probability","mean_validity","mean_novelty","mean_alignment"]
def scenario_label(v): return SCENARIO_LABELS.get(str(v),str(v).replace('_',' '))
def event_label(v): return EVENT_LABELS.get(str(v),str(v).replace('_',' '))
def phase_label(v): return PHASE_LABELS.get(str(v),str(v).replace('_',' '))
def metric_label(v): return METRIC_LABELS.get(str(v),str(v).replace('_',' '))
