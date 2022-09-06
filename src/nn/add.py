from . import Module
from .. import Tensor

from typing import Tuple


class Add(Module):
    """
    Addition operation between two Tensors.
    """
    def __init__(self) -> None:
        super(Add, self).__init__()

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        return x1 + x2

    def backward(self, dy: Tensor) -> Tuple[Tensor, Tensor]:
        return dy, dy
