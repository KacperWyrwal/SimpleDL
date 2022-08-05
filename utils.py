import numpy as np


def _as_pairs(padding, ndim):
    padding = np.asarray(padding)

    if padding.size == 1:
        padding = padding.ravel()
        return ((padding[0], padding[0]),) * ndim

    if padding.shape == (1, 2):
        return ((padding[0, 0], padding[0, 1]),) * ndim

    return padding


def unpad(arr, padding=0):
    padding = _as_pairs(padding, arr.ndim)
    slices = tuple(slice(l, s-r) for s, (l, r) in zip(arr.shape, padding))
    return arr[slices]


