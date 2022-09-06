from numpy import arange, asarray, setdiff1d, zeros

from typing import Union


def _strides_from_reduced_axis(
        strides: tuple[int, ...],
        axis: Union[None, int, tuple[int, ...]],
        fill: Union[int, tuple[int, ...]] = 0,
) -> tuple[int, ...]:
    """

    :param strides: Original strides of the array which has been reduced.
    :param axis: Axis over which the array has been reduced.
    :param fill: Fill value for strides of axes which have been reduced.
    :return:
    """
    n_dim = len(strides)

    # If axis is None, then the array has been reduced over all its axes
    if axis is None:
        axis = arange(n_dim)
    else:
        axis = asarray(axis)
        # Modulo needed for set difference with negative axes
        axis %= n_dim

    axis_not_reduced = setdiff1d(arange(n_dim), axis)

    # Fill the axis reduced over with the given fill value(s) and the rest with the original strides
    full_strides = zeros(n_dim, dtype=int)
    full_strides[axis] = fill
    full_strides[axis_not_reduced] = strides
    return tuple(full_strides)
