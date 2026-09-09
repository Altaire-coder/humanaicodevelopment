import numpy as np

def wright_fisher_sample(probabilities, population_size, rng):
    probabilities = np.asarray(probabilities, dtype=float)
    probabilities = probabilities / probabilities.sum()
    counts = rng.multinomial(population_size, probabilities)
    return counts / population_size
