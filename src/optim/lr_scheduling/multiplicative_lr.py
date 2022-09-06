from . import LRScheduler

from typing import Callable


class MultiplicativeLR(LRScheduler):

    def __init__(self, lr_lambda: Callable[[int], float], epoch: int = -1) -> None:
        super().__init__(epoch=epoch)
        self.lr_lambda = lr_lambda

    def _step(self, lr: float) -> float:
        return lr * self.lr_lambda(self.epoch)
