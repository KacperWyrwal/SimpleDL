import numpy as np


rng = np.random.default_rng()

def normal(shape, mean: float = 0.0, std: float = 1.0):
    return rng.normal(loc=mean, scale=std, size=shape)
