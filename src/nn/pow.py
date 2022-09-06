from . import Module
from .. import Tensor

# This will ideally be later changed to sth like 'from simpleDL import power' where simpleDL is just an alias for numpy
from numpy import power


class Pow(Module):
    """
    Raises a Tensor to a power.
    """
    def __init__(self) -> None:
        super(Pow, self).__init__()
        self.x = None
        self.p = None

    def forward(self, x: Tensor, p: float) -> Tensor:
        self.x = x
        self.p = p
        return power(x, p)

    def backward(self, dy: Tensor) -> Tensor:
        return (self.p - 1) * power(self.x, self.p - 1)
