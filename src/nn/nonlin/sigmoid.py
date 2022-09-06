from .. import Module
from ... import Tensor

from numpy import exp


class Sigmoid(Module):

    # TODO: Normally we would write this in terms of the simpler functions; however, in this case the factors cancel
    # TODO: out nicely. It the speed-up justified at the cost of more less visible chain rule?
    def __init__(self) -> None:
        super(Sigmoid, self).__init__()
        self.y = None

    def forward(self, x: Tensor) -> Tensor:
        self.y = 1 / (1 + exp(-x))
        return self.y

    def backward(self, dy: Tensor) -> Tensor:
        return self.y * (1 - self.y) * dy
