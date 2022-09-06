from . import Module, Add, Mul
from .. import Tensor

from typing import Tuple


class Sub(Module):
    """
    Subtraction operation between two Tensors.
    """
    def __init__(self) -> None:
        super(Sub, self).__init__()
        self.add = Add()
        self.mul = Mul()

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        x2 = self.mul(x2, -1)
        return self.add(x1, x2)

    def backward(self, dy: Tensor) -> Tuple[Tensor, Tensor]:
        dx1, dy2 = self.add.backward(dy)
        dx2, _ = self.mul.backward(dy2)
        return dx1, dx2
