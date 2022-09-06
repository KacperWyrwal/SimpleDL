from abc import ABC, abstractmethod


class LRScheduler(ABC):

    def __init__(self, epoch: int = -1) -> None:
        self.epoch = epoch

    def step(self, lr: float) -> float:
        self.epoch += 1
        if self.epoch == 0:
            return self._setup(lr=lr)
        else:
            return self._step(lr=lr)

    @abstractmethod
    def _step(self, lr: float) -> float:
        ...

    def _setup(self, lr: float) -> float:
        return lr
