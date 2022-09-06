from . import Module
from .. import Tensor

# Ideally numpy aliased as simple_dl or sth
from numpy import exp


class Exp(Module):
    """
    Exponentiates a Tensor element-wise.
    """
    def __init__(self) -> None:
        super(Exp, self).__init__()
        self.x = None

    def forward(self, x: Tensor) -> Tensor:
        self.x = x
        return exp(x)

    def backward(self, dy: Tensor) -> Tensor:
        return dy * exp(self.x)
