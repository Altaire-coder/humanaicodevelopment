from __future__ import annotations
import pandas as pd, matplotlib.pyplot as plt
from .labels import scenario_label
from .styles import save_figure

def plot_phase_diagram_faceted(outcome_metrics,output_dir,x='recursive_degradation_risk',y='co_development_score',sample_per_scenario=2500,vector_format='pdf'):
    if outcome_metrics.empty or x not in outcome_metrics or y not in outcome_metrics: return
    scenarios=list(outcome_metrics.scenario_id.drop_duplicates()); fig,axes=plt.subplots(2,2,figsize=(8.2,6.4),sharex=True,sharey=True); axes=axes.flatten(); xmin,xmax=outcome_metrics[x].min(),outcome_metrics[x].max(); ymin,ymax=outcome_metrics[y].min(),outcome_metrics[y].max()
    for ax,sid in zip(axes,scenarios):
        g=outcome_metrics[outcome_metrics.scenario_id==sid].copy();
        if len(g)>sample_per_scenario: g=g.sample(sample_per_scenario,random_state=42)
        ax.scatter(g[x],g[y],s=6,alpha=.20); ax.set_title(scenario_label(sid)); ax.set_xlim(xmin,xmax); ax.set_ylim(ymin,ymax)
    for ax in axes[len(scenarios):]: ax.axis('off')
    for ax in axes[::2]: ax.set_ylabel('Co-development')
    for ax in axes[-2:]: ax.set_xlabel('Recursive degradation risk')
    fig.suptitle('Co-development-recursive-degradation phase space by scenario'); save_figure(fig,output_dir,'fig_14_phase_diagram_faceted',vector_format=vector_format); plt.close(fig)
