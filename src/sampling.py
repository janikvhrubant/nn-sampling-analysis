import numpy as np
from scipy.stats import qmc

def generate_random_sequence(num_samples: int, dim: int):
    return np.random.rand(num_samples, dim)

def generate_sobol_sequence(num_samples: int, dim: int):
    sampler = qmc.Sobol(d=dim, scramble=False)
    num_samples_opt = 2**int(np.ceil(np.log2(num_samples)))
    samples = sampler.random(n=num_samples_opt)
    return samples[:num_samples, :]

def generate_halton_sequence(num_samples: int, dim: int):
    sampler = qmc.Halton(d=dim, scramble=False)
    return sampler.random(n=num_samples)
