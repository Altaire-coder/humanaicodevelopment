from __future__ import annotations
import pandas as pd, matplotlib.pyplot as plt
from .labels import MAIN_METRICS,metric_label,scenario_label
from .styles import save_figure

def plot_scenario_contrast_forest(scenario_contrasts,output_dir,metrics=None,baseline_scenario='S0_no_treatment',vector_format='pdf'):
    if scenario_contrasts.empty: return
    metrics=metrics or MAIN_METRICS
    d=scenario_contrasts[(scenario_contrasts.metric.isin(metrics)) & (scenario_contrasts.scenario_id!=baseline_scenario)].copy()
    if d.empty: return
    d['measure']=d.metric.map(metric_label); d['scenario']=d.scenario_id.map(scenario_label)
    metric_order=[metric_label(m) for m in metrics if m in set(d.metric)]
    scenarios=list(d.scenario.drop_duplicates()); ymap={m:i for i,m in enumerate(metric_order)}
    fig,ax=plt.subplots(figsize=(8,5.8)); step=0.18; center=(len(scenarios)-1)/2 if scenarios else 0
    for i,s in enumerate(scenarios):
        g=d[d.scenario==s]; y=[ymap[m]+(i-center)*step for m in g.measure]
        ax.errorbar(g['mean'],y,xerr=g['ci95'],fmt='o',capsize=3,label=s)
    ax.axvline(0,linestyle='--',linewidth=1); ax.set_yticks(list(ymap.values())); ax.set_yticklabels(metric_order); ax.set_xlabel('Difference from S0 No treatment'); ax.set_title('Scenario effects relative to no treatment'); ax.legend(frameon=False)
    save_figure(fig,output_dir,'fig_08_scenario_contrast_forest',vector_format=vector_format); plt.close(fig)

def plot_final_metric_bar(final_state_summary,output_dir,metric,vector_format='pdf'):
    if final_state_summary.empty: return
    d=final_state_summary[final_state_summary.metric==metric].copy()
    if d.empty: return
    d['scenario']=d.scenario_id.map(scenario_label); fig,ax=plt.subplots(figsize=(7,4)); ax.bar(d.scenario,d['mean'],yerr=d.ci95,capsize=3); ax.set_title(f'Final {metric_label(metric)} by scenario'); ax.set_ylabel(metric_label(metric)); ax.tick_params(axis='x',rotation=20)
    save_figure(fig,output_dir,f"fig_09_final_{metric.replace('/','_')}_bar",vector_format=vector_format); plt.close(fig)
