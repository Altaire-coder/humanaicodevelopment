from __future__ import annotations
from dataclasses import dataclass
from uuid import uuid4
import numpy as np
from .idea import Idea

@dataclass(frozen=True)
class MutationOutcome:
    child: Idea
    validity_delta: float
    novelty_delta: float
    alignment_delta: float
    error_delta: float

def classify_mutation(validity_delta, novelty_delta, alignment_delta, error_delta):
    if validity_delta >= 0.10 and error_delta <= -0.10:
        return 'corrective'
    if validity_delta <= -0.40 or alignment_delta <= -0.40:
        return 'catastrophic'
    quality_gain = validity_delta + alignment_delta - error_delta
    if quality_gain > 0.10 and novelty_delta >= 0:
        return 'beneficial'
    if quality_gain < -0.10 or error_delta > 0.10:
        return 'deleterious'
    return 'neutral'

def mutate_idea(*, parent: Idea, rng: np.random.Generator, time: int, secondary_parent: Idea | None = None, mutation_scale: float = 0.10, correction_bias: float = 0.0, distortion_bias: float = 0.0):
    if secondary_parent:
        base_validity = (parent.validity + secondary_parent.validity)/2
        base_novelty = (parent.novelty + secondary_parent.novelty)/2
        base_alignment = (parent.alignment + secondary_parent.alignment)/2
        base_error = (parent.error_severity + secondary_parent.error_severity)/2
        depth = max(parent.lineage_depth, secondary_parent.lineage_depth)
        hs = (parent.human_origin_share + secondary_parent.human_origin_share)/2
        ais = (parent.ai_origin_share + secondary_parent.ai_origin_share)/2
        es = (parent.external_origin_share + secondary_parent.external_origin_share)/2
        si = (parent.source_independence + secondary_parent.source_independence)/2
    else:
        base_validity, base_novelty, base_alignment, base_error = parent.validity, parent.novelty, parent.alignment, parent.error_severity
        depth = parent.lineage_depth
        hs, ais, es, si = parent.human_origin_share, parent.ai_origin_share, parent.external_origin_share, parent.source_independence
    validity = float(np.clip(base_validity + rng.normal(correction_bias-distortion_bias, mutation_scale),0,1))
    novelty = float(np.clip(base_novelty + rng.normal(0.025, mutation_scale),0,1))
    alignment = float(np.clip(base_alignment + rng.normal(correction_bias*0.5-distortion_bias, mutation_scale*0.75),0,1))
    error = float(np.clip(base_error + rng.normal(distortion_bias-correction_bias, mutation_scale),0,1))
    vd, nd, ad, ed = validity-base_validity, novelty-base_novelty, alignment-base_alignment, error-base_error
    mtype = classify_mutation(vd,nd,ad,ed)
    mag = float(np.clip((vd*vd+nd*nd+ad*ad+ed*ed)**0.5/2,0,1))
    utility = float(np.clip(0.45*validity+0.25*alignment+0.20*novelty-0.30*error,0,1))
    child = Idea(
        idea_id=f'I_{uuid4().hex[:12]}', lineage_id=parent.lineage_id, root_idea_id=parent.root_idea_id,
        generation=max(parent.generation, secondary_parent.generation if secondary_parent else parent.generation)+1,
        source_type='hybrid', validity=validity, novelty=novelty, alignment=alignment, error_severity=error,
        parent_id=parent.idea_id, secondary_parent_id=secondary_parent.idea_id if secondary_parent else None,
        mutation_type=mtype, mutation_magnitude=mag, utility=utility, confidence=(validity+parent.confidence)/2,
        creativity=novelty*validity*alignment, technical_rigor=validity*(1-error),
        repetition_score=float(np.clip(1-abs(novelty-base_novelty)-mag,0,1)),
        distortion_score=float(np.clip(0.55*(1-alignment)+0.45*error,0,1)),
        human_origin_share=hs, ai_origin_share=ais, external_origin_share=es, source_independence=si,
        lineage_depth=depth+1, birth_time=time
    )
    return MutationOutcome(child, vd, nd, ad, ed)
