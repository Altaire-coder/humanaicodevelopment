from __future__ import annotations
from pathlib import Path
import pandas as pd
from .labels import MAIN_METRICS,event_label,metric_label,phase_label,scenario_label
from .styles import ensure_dir

def _write_latex(df,path):
    try: path.write_text(df.to_latex(index=False,float_format='%.4f'),encoding='utf-8')
    except Exception: path.write_text(df.to_csv(index=False),encoding='utf-8')
def _save(df,out,name):
    if df is None or df.empty: return
    out=ensure_dir(out); df.to_csv(out/f'{name}.csv',index=False); _write_latex(df,out/f'{name}.tex')

def build_run_inventory(stage):
    e=stage.get('events',pd.DataFrame())
    if e.empty: return pd.DataFrame()
    cols=[c for c in ['scenario_id','run_id','matched_run_id','seed'] if c in e.columns]
    runs=e[cols].drop_duplicates().copy()
    agg={'run_id':'nunique'}
    out=runs.groupby('scenario_id',as_index=False).agg(runs=('run_id','nunique'))
    if 'matched_run_id' in runs: out['matched_runs']=runs.groupby('scenario_id')['matched_run_id'].nunique().values
    if 'seed' in runs:
        seed=runs.groupby('scenario_id')['seed'].agg(['min','max']).reset_index(); out=out.merge(seed,on='scenario_id',how='left').rename(columns={'min':'min_seed','max':'max_seed'})
    out.insert(1,'scenario',out['scenario_id'].map(scenario_label)); return out

def build_descriptive_metric_table(analysis):
    fs=analysis.get('final_state_summary',pd.DataFrame())
    if fs.empty: return pd.DataFrame()
    d=fs[fs.metric.isin(MAIN_METRICS)].copy(); d['scenario']=d.scenario_id.map(scenario_label); d['measure']=d.metric.map(metric_label)
    d['mean_ci95']=d.apply(lambda r:f"{r['mean']:.4f} ± {r['ci95']:.4f}",axis=1)
    return d.pivot_table(index=['metric','measure'],columns='scenario',values='mean_ci95',aggfunc='first').reset_index()

def build_scenario_contrast_table(analysis):
    c=analysis.get('scenario_contrasts',pd.DataFrame())
    if c.empty: return pd.DataFrame()
    d=c[(c.metric.isin(MAIN_METRICS)) & (c.scenario_id!=c.baseline_scenario)].copy(); d['scenario']=d.scenario_id.map(scenario_label); d['measure']=d.metric.map(metric_label)
    d['delta_ci95']=d.apply(lambda r:f"{r['mean']:.4f} ± {r['ci95']:.4f}",axis=1)
    return d.pivot_table(index=['metric','measure'],columns='scenario',values='delta_ci95',aggfunc='first').reset_index()

def build_intervention_summary_tables(stage,analysis):
    e=stage.get('events',pd.DataFrame()); timing=analysis.get('intervention_timing',pd.DataFrame())
    counts=pd.DataFrame()
    if not e.empty and 'is_intervention' in e:
        inter=e[e.is_intervention.fillna(False)].copy()
        if not inter.empty:
            base=e[['scenario_id','run_id']].drop_duplicates().groupby('scenario_id').run_id.nunique().rename('scenario_runs').reset_index()
            counts=inter.groupby(['scenario_id','event_type'],as_index=False).agg(total_events=('event_type','size'),runs_with_event=('run_id','nunique'),mean_event_time=('time','mean'),min_event_time=('time','min'),max_event_time=('time','max')).merge(base,on='scenario_id',how='left')
            counts['events_per_run']=counts.total_events/counts.scenario_runs; counts['scenario']=counts.scenario_id.map(scenario_label); counts['intervention']=counts.event_type.map(event_label)
            counts=counts[['scenario','intervention','total_events','scenario_runs','runs_with_event','events_per_run','mean_event_time','min_event_time','max_event_time']]
    ts=pd.DataFrame()
    if not timing.empty:
        t=timing.copy(); t['scenario']=t.scenario_id.map(scenario_label); t['intervention']=t.event_type.map(event_label)
        ts=t.groupby(['scenario','intervention'],as_index=False).agg(runs_with_event=('run_id','nunique'),mean_first_time=('first_time','mean'),sd_first_time=('first_time','std'),min_first_time=('first_time','min'),max_first_time=('first_time','max'),mean_event_count=('count','mean'))
    return counts,ts

def build_phase_summary_table(analysis):
    p=analysis.get('phase_share',pd.DataFrame())
    if p.empty: return pd.DataFrame()
    d=p.copy(); d['scenario']=d.scenario_id.map(scenario_label); d['phase_label']=d.phase.map(phase_label)
    return d.pivot_table(index='scenario',columns='phase_label',values='share',fill_value=0).reset_index()

def export_descriptive_tables(stage,analysis,output_dir):
    out=ensure_dir(Path(output_dir)); _save(build_run_inventory(stage),out,'table_00_run_inventory'); _save(build_descriptive_metric_table(analysis),out,'table_01_descriptive_final_metrics'); _save(build_scenario_contrast_table(analysis),out,'table_02_scenario_effects_vs_s0'); _save(build_phase_summary_table(analysis),out,'table_03_phase_summary')
    c,t=build_intervention_summary_tables(stage,analysis); _save(c,out,'table_04_intervention_frequency'); _save(t,out,'table_05_intervention_timing')
