import numpy as np

from . import Tensor


rng = np.random.default_rng()


def normal(shape, mean: float = 0.0, std: float = 1.0) -> Tensor:
    return rng.normal(loc=mean, scale=std, size=shape).view(Tensor)


def zeros(shape, dtype=float, order='C', *, like=None) -> Tensor:
    return np.zeros(shape=shape, dtype=dtype, order=order, like=like).view(Tensor)
