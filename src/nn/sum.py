from __future__ import annotations

from . import Module
from .. import Tensor
from .utils import _strides_from_reduced_axis

# ideally numpy aliased as simple_dl
from numpy import sum
from numpy.lib.stride_tricks import as_strided

from typing import Union


class Sum(Module):

    def __init__(self, axis: Union[None, int, tuple[int, ...]] = None) -> None:
        super(Sum, self).__init__()
        self.axis = axis
        self.x_shape = None

    def forward(self, x: Tensor) -> Tensor:
        self.x_shape = x.shape
        return sum(x, axis=self.axis)

    def backward(self, dy: Tensor) -> Tensor:
        return as_strided(
            dy,
            self.x_shape,
            _strides_from_reduced_axis(strides=dy.strides, axis=self.axis, fill=0),
            writeable=False,
        )
