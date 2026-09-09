from __future__ import annotations
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from .labels import event_label,scenario_label
from .styles import save_figure

def plot_intervention_frequency_per_run(events,output_dir,vector_format='pdf'):
    if events.empty or 'is_intervention' not in events: return
    inter=events[events.is_intervention.fillna(False)].copy()
    if inter.empty: return
    base=events[['scenario_id','run_id']].drop_duplicates(); rc=inter.groupby(['scenario_id','run_id','event_type']).size().rename('count').reset_index(); et=rc[['scenario_id','event_type']].drop_duplicates(); exp=base.merge(et,on='scenario_id',how='left').merge(rc,on=['scenario_id','run_id','event_type'],how='left'); exp['count']=exp['count'].fillna(0)
    s=exp.groupby(['scenario_id','event_type'],as_index=False).agg(mean_count=('count','mean'),std_count=('count','std'),n=('run_id','nunique')); s['ci95']=1.96*s.std_count.fillna(0)/np.sqrt(s.n.clip(lower=1)); s['label']=s.scenario_id.map(scenario_label)
    fig,ax=plt.subplots(figsize=(7.4,4.4)); ax.bar(s.label,s.mean_count,yerr=s.ci95,capsize=3); ax.set_title('Intervention frequency per run'); ax.set_ylabel('Mean count per run'); ax.tick_params(axis='x',rotation=20); save_figure(fig,output_dir,'fig_12_intervention_frequency_per_run',vector_format=vector_format); plt.close(fig)

def plot_first_intervention_timing_strip(intervention_timing,output_dir,vector_format='pdf'):
    if intervention_timing.empty or 'first_time' not in intervention_timing: return
    d=intervention_timing.copy(); d['intervention']=d.event_type.map(event_label); order=list(d.intervention.drop_duplicates()); ypos={n:i for i,n in enumerate(order)}; fig,ax=plt.subplots(figsize=(7.4,3.8)); rng=np.random.default_rng(42)
    for name,g in d.groupby('intervention',sort=False):
        y=ypos[name]; ax.scatter(g.first_time,np.full(len(g),y)+rng.normal(0,.045,len(g)),alpha=.65,s=24); ax.plot([g.first_time.median(),g.first_time.median()],[y-.25,y+.25],linewidth=2)
    ax.set_yticks(list(ypos.values())); ax.set_yticklabels(order); ax.set_xlabel('First intervention time'); ax.set_title('First intervention timing by policy'); save_figure(fig,output_dir,'fig_13_first_intervention_timing_strip',vector_format=vector_format); plt.close(fig)
