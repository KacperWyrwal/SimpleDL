from . import Module, Mul, Pow
from .. import Tensor

from typing import Tuple


class Div(Module):
    """
    Divides one Tensor by another element-wise.
    """
    def __init__(self) -> None:
        super(Div, self).__init__()
        self.mul = Mul()
        self.pow = Pow()

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        x2 = self.pow(x2, -1)
        return self.mul(x1, x2)

    def backward(self, dy: Tensor) -> Tuple[Tensor, Tensor]:
        dx1, dy2 = self.mul.backward(dy)
        dx2, _ = self.pow.backward(dy)
        return dx1, dx2
