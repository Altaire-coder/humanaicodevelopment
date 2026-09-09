from __future__ import annotations
from collections import Counter, defaultdict
from dataclasses import dataclass
import networkx as nx
import numpy as np
import pandas as pd
from .idea import Idea
from .lineage import Lineage, LineageEvent

@dataclass(frozen=True)
class GenealogyMetrics:
    active_ideas:int; extinct_ideas:int; active_lineages:int; extinct_lineages:int
    idea_survival_rate:float; valid_idea_survival_rate:float
    idea_vanishing_rate:float; valid_idea_vanishing_rate:float
    lineage_concentration:float; effective_lineage_diversity:float
    mutation_rate:float; beneficial_mutation_rate:float; corrective_mutation_rate:float; distortion_rate:float
    valid_reproduction_number:float; error_reproduction_number:float; novel_reproduction_number:float

class IdeaGenealogy:
    def __init__(self):
        self.ideas:dict[str,Idea]={}; self.lineages:dict[str,Lineage]={}; self.events:list[LineageEvent]=[]; self.graph=nx.DiGraph()

    def add_root(self, idea:Idea, *, run_id:str, scenario_id:str, time:int, source_id:str):
        if idea.idea_id in self.ideas or idea.parent_id is not None: raise ValueError('Invalid root idea.')
        self.ideas[idea.idea_id]=idea
        self.lineages[idea.lineage_id]=Lineage(idea.lineage_id, idea.idea_id, time, [idea.idea_id])
        self.graph.add_node(idea.idea_id, **idea.to_record())
        ev=LineageEvent.create(run_id=run_id,scenario_id=scenario_id,time=time,generation=idea.generation,event_type='birth',lineage_id=idea.lineage_id,idea_id=idea.idea_id,parent_idea_id=None,secondary_parent_id=None,source_id=source_id,target_id=idea.idea_id,mutation_type=idea.mutation_type,validity=idea.validity,error_severity=idea.error_severity,alive=True)
        self.events.append(ev); self.lineages[idea.lineage_id].event_ids.append(ev.event_id); return ev

    def add_descendant(self, idea:Idea, *, run_id:str, scenario_id:str, time:int, source_id:str, event_type:str|None=None):
        if idea.parent_id not in self.ideas: raise ValueError('Parent must exist.')
        if idea.secondary_parent_id and idea.secondary_parent_id not in self.ideas: raise ValueError('Secondary parent must exist.')
        if idea.lineage_id not in self.lineages: raise ValueError('Lineage must exist.')
        parent=self.ideas[idea.parent_id]
        if idea.generation<=parent.generation or idea.lineage_depth<=parent.lineage_depth: raise ValueError('Invalid generation/depth.')
        self.ideas[idea.idea_id]=idea; self.lineages[idea.lineage_id].idea_ids.append(idea.idea_id)
        parent.descendant_count+=1; self._increment_ancestors(parent.idea_id)
        self.graph.add_node(idea.idea_id, **idea.to_record()); self.graph.add_edge(parent.idea_id,idea.idea_id,edge_type='idea_inheritance',mutation_type=idea.mutation_type)
        if idea.secondary_parent_id:
            self.ideas[idea.secondary_parent_id].descendant_count+=1; self._increment_ancestors(idea.secondary_parent_id)
            self.graph.add_edge(idea.secondary_parent_id,idea.idea_id,edge_type='idea_recombination',mutation_type=idea.mutation_type)
        if event_type is None:
            event_type='recombination' if idea.secondary_parent_id else ('correction' if idea.mutation_type=='corrective' else ('distortion' if idea.mutation_type in {'deleterious','catastrophic'} else 'reproduction'))
        ev=LineageEvent.create(run_id=run_id,scenario_id=scenario_id,time=time,generation=idea.generation,event_type=event_type,lineage_id=idea.lineage_id,idea_id=idea.idea_id,parent_idea_id=idea.parent_id,secondary_parent_id=idea.secondary_parent_id,source_id=source_id,target_id=idea.idea_id,mutation_type=idea.mutation_type,validity=idea.validity,error_severity=idea.error_severity,alive=idea.alive)
        self.events.append(ev); self.lineages[idea.lineage_id].event_ids.append(ev.event_id); return ev

    def mark_extinct(self, idea_id:str, *, run_id:str, scenario_id:str, time:int, source_id:str='system'):
        idea=self.ideas[idea_id]; idea.mark_extinct(time); self.graph.nodes[idea_id].update(idea.to_record())
        ev=LineageEvent.create(run_id=run_id,scenario_id=scenario_id,time=time,generation=idea.generation,event_type='extinction',lineage_id=idea.lineage_id,idea_id=idea.idea_id,parent_idea_id=idea.parent_id,secondary_parent_id=idea.secondary_parent_id,source_id=source_id,target_id=idea.idea_id,mutation_type=idea.mutation_type,validity=idea.validity,error_severity=idea.error_severity,alive=False)
        self.events.append(ev); self.lineages[idea.lineage_id].event_ids.append(ev.event_id)
        if not any(x.alive and x.lineage_id==idea.lineage_id for x in self.ideas.values()): self.lineages[idea.lineage_id].alive=False; self.lineages[idea.lineage_id].extinction_time=time
        return ev

    def reintroduce(self, idea_id:str, *, run_id:str, scenario_id:str, time:int, source_id:str='external_data'):
        idea=self.ideas[idea_id]; idea.reintroduce(time); lin=self.lineages[idea.lineage_id]; lin.alive=True; lin.extinction_time=None; lin.reintroduction_count+=1
        ev=LineageEvent.create(run_id=run_id,scenario_id=scenario_id,time=time,generation=idea.generation,event_type='reintroduction',lineage_id=idea.lineage_id,idea_id=idea.idea_id,parent_idea_id=idea.parent_id,secondary_parent_id=idea.secondary_parent_id,source_id=source_id,target_id=idea.idea_id,mutation_type=idea.mutation_type,validity=idea.validity,error_severity=idea.error_severity,alive=True,is_intervention=True)
        self.events.append(ev); lin.event_ids.append(ev.event_id); return ev

    def _increment_ancestors(self, idea_id:str):
        current=idea_id
        while current:
            self.ideas[current].cumulative_descendants+=1; current=self.ideas[current].parent_id

    def node_records(self):
        rows=[]
        for i in self.ideas.values():
            status='extinct' if not i.alive else ('corrected' if i.mutation_type=='corrective' else ('distorted' if i.mutation_type in {'deleterious','catastrophic'} else ('erroneous' if i.is_erroneous else ('repetitive' if i.repetition_score>=.8 else ('novel_active' if i.novelty>=.7 else 'valid_active')))))
            rows.append({'node_id':i.idea_id,'node_type':'idea','label':i.idea_id,'status':status,'validity':i.validity,'novelty':i.novelty,'alignment':i.alignment,'utility':i.utility,'confidence':i.confidence,'error_severity':i.error_severity,'lineage_id':i.lineage_id,'generation':i.generation,'alive':i.alive,'size_value':max(1,i.cumulative_descendants+1),'opacity_value':1.0 if i.alive else .2})
        return pd.DataFrame(rows)

    def edge_records(self):
        return pd.DataFrame([{'source_id':u,'target_id':v,'edge_type':d.get('edge_type'),'idea_id':v,'lineage_id':self.ideas[v].lineage_id,'validity':self.ideas[v].validity,'mutation_type':self.ideas[v].mutation_type,'alive':self.ideas[v].alive} for u,v,d in self.graph.edges(data=True)])
    def lineage_records(self): return pd.DataFrame([i.to_record() for i in self.ideas.values()])
    def event_records(self): return pd.DataFrame([e.to_record() for e in self.events])

    def compute_metrics(self, prior_active_ids=None, novelty_threshold=.7):
        ideas=list(self.ideas.values()); active=[i for i in ideas if i.alive]; extinct=[i for i in ideas if not i.alive]
        prior=set(prior_active_ids or []); current={i.idea_id for i in active}; vanished=prior-current
        sr=len(prior&current)/len(prior) if prior else 1.; vr=len(vanished)/len(prior) if prior else 0.
        pv={x for x in prior if x in self.ideas and self.ideas[x].is_valid}; cv={i.idea_id for i in active if i.is_valid}
        vs=len(pv&cv)/len(pv) if pv else 1.; vv=len(pv-cv)/len(pv) if pv else 0.
        counts=Counter(i.lineage_id for i in active); total=max(1,len(active)); p=np.array([c/total for c in counts.values()]); lc=float((p**2).sum()) if len(p) else 0.; eld=0 if lc==0 else 1/lc
        desc=defaultdict(list)
        for i in ideas:
            if i.parent_id: desc[i.parent_id].append(i)
        vrc=[len(desc[i.idea_id]) for i in ideas if i.is_valid]; erc=[len(desc[i.idea_id]) for i in ideas if i.is_erroneous]; nrc=[len(desc[i.idea_id]) for i in ideas if i.novelty>=novelty_threshold]
        mutated=[i for i in ideas if i.mutation_type!='none']; m=max(1,len(mutated))
        return GenealogyMetrics(len(active),len(extinct),sum(l.alive for l in self.lineages.values()),sum(not l.alive for l in self.lineages.values()),sr,vs,vr,vv,lc,eld,len(mutated)/max(1,len(ideas)),sum(i.mutation_type=='beneficial' for i in mutated)/m,sum(i.mutation_type=='corrective' for i in mutated)/m,sum(i.mutation_type in {'deleterious','catastrophic'} for i in mutated)/m,float(np.mean(vrc)) if vrc else 0.,float(np.mean(erc)) if erc else 0.,float(np.mean(nrc)) if nrc else 0.)

    @classmethod
    def from_stage2(cls, idea_states:pd.DataFrame, events:pd.DataFrame|None=None):
        g=cls(); rows=idea_states.sort_values(['time','generation']).to_dict('records')
        for r in rows:
            idea=Idea(idea_id=r['idea_id'],parent_id=r.get('parent_idea_id'),secondary_parent_id=r.get('secondary_parent_id'),lineage_id=r['lineage_id'],root_idea_id=r.get('root_idea_id') or (r.get('parent_idea_id') or r['idea_id']),generation=int(r['generation']),source_type=r.get('origin_type','hybrid'),mutation_type=r.get('mutation_type','neutral'),mutation_magnitude=float(r.get('mutation_magnitude',0)),validity=float(r['validity']),novelty=float(r['novelty']),alignment=float(r['alignment']),error_severity=float(r['error_severity']),utility=float(r.get('utility',0)),confidence=float(r.get('confidence',0)),creativity=float(r.get('creativity',0)),technical_rigor=float(r.get('technical_rigor',0)),repetition_score=float(r.get('repetition_score',0)),distortion_score=float(r.get('distortion_score',0)),human_origin_share=float(r.get('human_origin_share',0)),ai_origin_share=float(r.get('ai_origin_share',0)),external_origin_share=float(r.get('external_origin_share',0)),source_independence=float(r.get('source_independence',0)),lineage_depth=int(r.get('lineage_depth',r['generation'])),alive=bool(r.get('alive',True)),birth_time=int(r.get('time',0)))
            if idea.parent_id is None or idea.parent_id not in g.ideas: g.add_root(idea,run_id=r.get('run_id','R'),scenario_id=r.get('scenario_id','S'),time=int(r.get('time',0)),source_id='stage2_import')
            else: g.add_descendant(idea,run_id=r.get('run_id','R'),scenario_id=r.get('scenario_id','S'),time=int(r.get('time',0)),source_id='stage2_import')
        return g
