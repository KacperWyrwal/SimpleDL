from . import LRScheduler

from typing import Callable


class LambdaLR(LRScheduler):
    """
    Always returns initial learning rate times lr_lambda(epoch).
    """

    def __init__(self, lr_lambda: Callable[[int], float], epoch: int = -1) -> None:
        super().__init__(epoch=epoch)
        self.lr_lambda = lr_lambda
        self.base_lr = None

    def _step(self, lr: float) -> float:
        return self.base_lr * self.lr_lambda(self.epoch)

    def _setup(self, lr: float) -> float:
        self.base_lr = lr
        return lr
