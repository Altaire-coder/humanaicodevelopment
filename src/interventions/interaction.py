import numpy as np

def _clip(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def c(x): return float(np.clip(x,0,1))

def apply_human_data_injection(human_input, injection_share, injected_validity, injected_novelty, injected_alignment, source_independence):
    lam = c(injection_share)
    out = dict(human_input)
    out["validity"] = c((1-lam)*out["validity"] + lam*injected_validity)
    out["novelty"] = c((1-lam)*out["novelty"] + lam*injected_novelty)
    out["alignment"] = c((1-lam)*out["alignment"] + lam*injected_alignment)
    out["human_origin_share"] = c(out.get("human_origin_share",0)+lam*source_independence)
    out["ai_origin_share"] = c(out.get("ai_origin_share",0)*(1-lam))
    out["source_independence"] = c(source_independence)
    return out

def apply_critic_feedback(ai_output, critic_strength, counterfactual_strength, uncertainty_flagging, intention_anchor_strength):
    out = dict(ai_output)
    out["novelty"] = c(out["novelty"] + .25*critic_strength + .2*counterfactual_strength)
    out["alignment"] = c(out["alignment"] + .3*intention_anchor_strength)
    out["validity"] = c(out["validity"] + .1*critic_strength)
    out["confidence"] = c(out["confidence"] - .25*uncertainty_flagging)
    out["agreement"] = c(out.get("agreement",.5) - .3*critic_strength)
    return out

def apply_context_reset(
    *,
    ai_state,
    reset_strength: float,
    preserve_verified_summary: bool,
) -> None:
    strength = _clip(reset_strength)

    ai_state.context_contamination *= (
        1.0 - strength
    )
    ai_state.memory_strength *= (
        1.0 - 0.75 * strength
    )

    if preserve_verified_summary:
        ai_state.local_alignment_conditioning = _clip(
            ai_state.local_alignment_conditioning
            + 0.10 * strength
        )
        ai_state.current_intention_anchor = _clip(
            ai_state.current_intention_anchor
            + 0.10 * strength
        )