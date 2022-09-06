from abc import ABC

from . import Optimizer
from .. import Parameter
from lr_scheduling import LRScheduler
from functools import reduce

from typing import Union, Iterator, Tuple, Any


class LROptimizer(Optimizer, ABC):

    def __init__(self, params: Union[Iterator[Tuple[str, Parameter]], dict[str, Any]], lr: float,
                 lr_schedulers: Union[list[LRScheduler], LRScheduler, None] = None,
                 **defaults) -> None:
        super().__init__(params=params, lr=lr, lr_schedulers=lr_schedulers, **defaults)
        self._prep_lr_schedulers()
        self.schedule_lr()

    def _prep_lr_schedulers(self) -> None:
        """
        Ensures that the 'lr_schedulers' attributes of each param_group is a list.
        """
        for param_group in self.param_groups:
            lr_schedulers = param_group['lr_schedulers']
            # TODO another solution would be to have an identity scheduler class and set it here
            if lr_schedulers is None:
                continue
            elif isinstance(lr_schedulers, LRScheduler):
                lr_schedulers = [lr_schedulers]
            else:
                lr_schedulers = list(lr_schedulers)
            param_group['lr_schedulers'] = lr_schedulers

    def schedule_lr(self) -> None:
        """
        Applies all lr_schedulers in a given param_group to the learning rate of that group.
        """
        for param_group in self.param_groups:
            lr_schedulers = param_group['lr_schedulers']
            # It is possible that one param group has a scheduler, while another does not
            if lr_schedulers is not None:
                param_group['lr'] = reduce(
                    lambda lr, lr_scheduler: lr_scheduler.step(lr),
                    param_group['lr_schedulers'],
                    param_group['lr'],
                )