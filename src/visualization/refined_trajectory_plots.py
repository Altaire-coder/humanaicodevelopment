from __future__ import annotations
import pandas as pd, matplotlib.pyplot as plt
from .labels import metric_label,scenario_label
from .styles import save_figure

def plot_scenario_trajectories_refined(trajectory_summary,output_dir,metric,burn_in=5,vector_format='pdf'):
    if trajectory_summary.empty: return
    d=trajectory_summary[(trajectory_summary.metric==metric)&(trajectory_summary.time>=burn_in)].copy()
    if d.empty: return
    fig,ax=plt.subplots(figsize=(7.2,4.4))
    for sid,g in d.groupby('scenario_id',sort=True):
        g=g.sort_values('time'); ax.plot(g.time,g['mean'],label=scenario_label(sid))
        if 'ci95' in g: ax.fill_between(g.time,g['mean']-g.ci95,g['mean']+g.ci95,alpha=.10)
    ax.set_title(f'Scenario trajectories after burn-in: {metric_label(metric)}'); ax.set_xlabel('Time'); ax.set_ylabel(metric_label(metric)); ax.legend(frameon=False)
    save_figure(fig,output_dir,f"fig_10_trajectory_burnin_{metric.replace('/','_')}",vector_format=vector_format); plt.close(fig)

def plot_error_reproduction_comparison(trajectory_summary,output_dir,burn_in=5,vector_format='pdf'):
    plot_scenario_trajectories_refined(trajectory_summary,output_dir,'R_E',burn_in,vector_format)
