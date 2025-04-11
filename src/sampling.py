import numpy as np
import sobol_seq

def generate_random_sequence(num_samples: int, dim: int):
    return np.random.rand(num_samples, dim)

def generate_sobol_sequence(num_samples: int, dim: int):
    return sobol_seq.i4_sobol_generate(dim, num_samples)

# def generate_