import numpy as np
def c(x): return float(np.clip(x,0,1))

def apply_verified_retraining(platform_state, ai_state, verified_training_share, human_training_share, contaminated_training_share, learning_rate):
    lr = c(learning_rate)
    quality = .7*c(verified_training_share) + .3*c(human_training_share)
    platform_state.model_contamination = c(
        (1-lr)*platform_state.model_contamination
        + lr*c(contaminated_training_share)
        - lr*.6*c(verified_training_share)
    )
    platform_state.response_diversity = c(
        platform_state.response_diversity + lr*.25*c(human_training_share)
    )
    platform_state.performance = c(
        platform_state.performance + lr*quality - lr*platform_state.model_contamination
    )
    ai_state.accuracy = c(
        ai_state.accuracy + lr*quality - lr*platform_state.model_contamination
    )
    ai_state.model_contamination = platform_state.model_contamination
