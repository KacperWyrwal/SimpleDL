from .. import Module, Sub, Pow, Mean
from ... import Tensor


class MSELoss(Module):

    # TODO: might need Mean axis parameter different than None to work with batches
    def __init__(self) -> None:
        super(MSELoss, self).__init__()
        self.sub = Sub()
        self.pow = Pow()
        self.mean = Mean()

    def forward(self, x: Tensor, x_hat: Tensor) -> Tensor:
        y = self.sub(x, x_hat)
        y = self.pow(y, 2)
        y = self.mean(y)
        return y

    def backward(self, dy: Tensor) -> Tensor:
        dy = self.mean.backward(dy)
        dy = self.pow.backward(dy)
        _, dx = self.sub.backward(dy)
        return dx
