import numpy as np
import sobol_seq

def generate_random_sequence(num_samples, dim):
    """
    Generate random samples using Random sampling.

    Args:
        num_samples (int): The number of samples to generate.
        dim (int): The dimensionality of the sample space.

    Returns:
        np.ndarray: A 2D array of generated samples.
    """
    return np.random.rand(num_samples, dim)

def generate_sobol_sequence(num_samples, dim):
    """
    Generate a Sobol sequence for quasi-random sampling.

    Args:
        num_samples (int): The number of samples to generate.
        dim (int): The dimensionality of the sequence.

    Returns:
        np.ndarray: A 2D array representing the Sobol sequence.
    """
    return sobol_seq.i4_sobol_generate(dim, num_samples)
