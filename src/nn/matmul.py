from . import Module
from .. import Tensor

from typing import Tuple


class MatMul(Module):
    """
    Matrix multiplication of two Tensors.
    """
    def __init__(self):
        super(MatMul, self).__init__()
        self.x1 = None
        self.x2 = None

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        self.x1 = x1
        self.x2 = x2
        return self.x1 @ self.x2

    def backward(self, dy: Tensor) -> Tuple[Tensor, Tensor]:
        dx1 = dy @ self.x2.T
        dx2 = self.x1.T @ dy
        return dx1, dx2
