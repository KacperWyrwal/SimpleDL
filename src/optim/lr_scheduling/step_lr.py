from . import LRScheduler


class StepLR(LRScheduler):

    # TODO perhaps rename step_size to step_epochs
    def __init__(self, step_size: int, gamma: float = 0.1, epoch: int = -1) -> None:
        super().__init__(epoch=epoch)
        self.step_size = step_size
        self.gamma = gamma

    def _step(self, lr: float) -> float:
        if self.epoch % self.step_size == 0:
            return lr * self.gamma
        return lr
