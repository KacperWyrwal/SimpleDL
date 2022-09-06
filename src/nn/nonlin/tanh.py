from .. import Module
from ... import Tensor

# Ideally aliased in the future
from numpy import tanh


class Tanh(Module):

    def __init__(self) -> None:
        super(Tanh, self).__init__()
        self.y = None

    def forward(self, x: Tensor) -> Tensor:
        self.y = tanh(x)
        return self.y

    def backward(self, dy: Tensor) -> Tensor:
        return (1 - self.y ** 2) * dy
