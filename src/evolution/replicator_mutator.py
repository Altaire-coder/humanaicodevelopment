import numpy as np

def update_frequencies(x, fitness, mutation_matrix):
    # x_j(t+1) = sum_i x_i f_i Q_ij / mean_fitness
    x = np.asarray(x, dtype=float)
    fitness = np.asarray(fitness, dtype=float)
    q = np.asarray(mutation_matrix, dtype=float)

    weighted = x * fitness
    denom = weighted.sum()
    if denom <= 0:
        raise ValueError("Mean fitness must be positive.")

    next_x = weighted @ q / denom
    next_x = np.clip(next_x, 0.0, None)
    return next_x / next_x.sum()
