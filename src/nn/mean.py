from . import Module, Sum, Div
from .. import Tensor
from math import prod


class Mean(Module):

    def __init__(self, axis=None):
        super(Mean, self).__init__()
        self.axis = axis
        self.sum = Sum(axis=axis)
        self.div = Div()

    def _number_of_elements_in_axis(self, x_shape):
        # TODO: Maybe add this function to utils
        return prod(map(x_shape.__getitem__, self.axis))

    def forward(self, x: Tensor) -> Tensor:
        return self.div(self.sum(x), self._number_of_elements_in_axis(x.shape))

    def backward(self, dy: Tensor) -> Tensor:
        dy, _ = self.div.backward(dy)
        dx = self.sum.backward(dy)
        return dx
