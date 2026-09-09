from __future__ import annotations
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch,FancyBboxPatch
from .styles import save_figure

def _box(ax,xy,text,w=2.5,h=.75):
    x,y=xy; p=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.05',linewidth=1.2,fill=False); ax.add_patch(p); ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=9)
def _arrow(ax,s,e): ax.add_patch(FancyArrowPatch(s,e,arrowstyle='->',mutation_scale=12,linewidth=1.2))
def plot_conceptual_scheme(output_dir,vector_format='pdf'):
    fig,ax=plt.subplots(figsize=(9.5,5.2)); ax.set_axis_off();
    _box(ax,(.2,3.4),'Human input\nvalidity / novelty / alignment'); _box(ax,(3.3,3.4),'AI response\naccuracy / reuse / confidence'); _box(ax,(6.4,3.4),'Human evaluation\nverification / reliance'); _box(ax,(3.3,2.1),'Revised idea\nmutation / reproduction'); _box(ax,(.2,.8),'Genealogy state\nvalid and error lineages'); _box(ax,(3.3,.8),'Recursive metrics\nco-development / risk'); _box(ax,(6.4,.8),'Scenario intervention\nS1 / S2 / S3')
    for s,e in [((2.7,3.78),(3.3,3.78)),((5.8,3.78),(6.4,3.78)),((7.65,3.4),(4.55,2.85)),((3.3,2.45),(2.7,1.18)),((2.7,1.18),(3.3,1.18)),((5.8,1.18),(6.4,1.18)),((6.4,1.55),(5.4,3.4)),((4.55,2.1),(1.45,3.4))]: _arrow(ax,s,e)
    ax.set_xlim(0,9.2); ax.set_ylim(.3,4.5); ax.set_title('Conceptual scheme of recursive human-AI co-development'); save_figure(fig,output_dir,'fig_00_conceptual_scheme',vector_format=vector_format); plt.close(fig)
def plot_intervention_mechanism_scheme(output_dir,vector_format='pdf'):
    fig,ax=plt.subplots(figsize=(9.5,4.6)); ax.set_axis_off();
    for xy,t,w in [((.2,2.8),'S0\nNo treatment',1.8),((2.4,2.8),'S1\nHuman data injection',2),((4.9,2.8),'S2\nCritic feedback',2),((7.4,2.8),'S3\nContext reset',2),((.2,1.3),'Baseline recursive\ninteraction',1.8),((2.4,1.3),'External human-origin\ninformation',2),((4.9,1.3),'Challenge / uncertainty\n/ counterfactual',2),((7.4,1.3),'Remove contaminated\nlocal memory',2),((3.3,.1),'Expected effect:\nquality improvement, risk reduction, or trade-off',3.4)]: _box(ax,xy,t,w=w,h=.8 if xy==(3.3,.1) else .75)
    for s,e in [((1.1,2.8),(1.1,2.05)),((3.4,2.8),(3.4,2.05)),((5.9,2.8),(5.9,2.05)),((8.4,2.8),(8.4,2.05)),((1.1,1.3),(3.3,.5)),((3.4,1.3),(4.3,.9)),((5.9,1.3),(5.7,.9)),((8.4,1.3),(6.7,.5))]: _arrow(ax,s,e)
    ax.set_xlim(0,9.8); ax.set_ylim(0,3.9); ax.set_title('Scenario intervention mechanisms'); save_figure(fig,output_dir,'fig_00b_intervention_mechanism_scheme',vector_format=vector_format); plt.close(fig)
