from . import Module
from .. import Tensor

from typing import Tuple

class Mul(Module):
    """
    Element-wise multiplication of two Tensors.
    """
    def __init__(self) -> None:
        super(Mul, self).__init__()
        self.x1 = None
        self.x2 = None

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        self.x1 = x1
        self.x2 = x2
        return x1 * x2

    def backward(self, dy: Tensor) -> Tuple[Tensor, Tensor]:
        return self.x2 * dy, self.x1 * dy