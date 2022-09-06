from . import LROptimizer
from .. import Tensor, Parameter
from lr_scheduling import LRScheduler

from typing import Union


class SGD(LROptimizer):
    def __init__(self, params, lr: float,
                 lr_schedulers: Union[list[LRScheduler], LRScheduler, None] = None,
                 momentum: float = 0, dampening: float = 0, weight_decay: float = 0,
                 nesterov: bool = False, *, maximize: bool = False):
        super().__init__(params=params, lr=lr, lr_schedulers=lr_schedulers, momentum=momentum,
                         dampening=dampening, weight_decay=weight_decay, nesterov=nesterov,
                         maximize=maximize)
        # store exponential moving average of the gradient if using momentum
        for param_group in self.param_groups:
            param_group['grads_ema'] = [param.grad if momentum != 0 else None
                                        for param in param_group['params']]

    def step(self):
        for param_group in self.param_groups:
            lr, momentum, dampening, weight_decay, nesterov, maximize = param_group['lr'], \
                                                                        param_group['momentum'], param_group[
                                                                            'dampening'], param_group['weight_decay'], \
                                                                        param_group['nesterov'], param_group['maximize']
            for param, grad_ema in zip(param_group['params'], param_group['grads_ema']):
                SGD.sgd(param=param, grad_ema=grad_ema, lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov, maximize=maximize)

    @staticmethod
    def sgd(param: Parameter, grad_ema: Tensor, lr: float, momentum: float = 0, dampening: float = 0,
            weight_decay: float = 0, nesterov: bool = False, *, maximize: bool = False) -> None:
        """
        Applies Stochastic Gradient Descent (or Ascent) on a given Parameter.

        TODO: consider whether copy should be optional to save space.
        """
        grad = param.grad.copy()
        # L2 penalty
        if weight_decay != 0:
            grad += weight_decay * param
            # momentum
        if momentum != 0:
            grad_ema = momentum * grad_ema + (1 - dampening) * grad
            # Adjustment for Nesterov's version of momentum SGD
        if nesterov is True:
            grad += momentum * grad_ema
            # Apply learning rate
        param.optimize(grad * lr, maximize=maximize)
