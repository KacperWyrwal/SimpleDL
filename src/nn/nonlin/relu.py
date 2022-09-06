from .. import Module
from ... import Tensor


class ReLU(Module):
    """
    Rectified Linear Unit.
    """
    def __init__(self) -> None:
        super(ReLU, self).__init__()
        self.x_gt_0 = None

    def forward(self, x: Tensor) -> Tensor:
        self.x_gt_0 = 0 < x
        return x * self.x_gt_0

    def backward(self, dy: Tensor) -> Tensor:
        return self.x_gt_0 * dy
