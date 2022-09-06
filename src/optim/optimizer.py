from .. import Parameter
from abc import ABC, abstractmethod

from typing import Union, Iterator, Tuple, Any


class Optimizer(ABC):

    def __init__(self, params: Union[Iterator[Tuple[str, Parameter]], dict[str, Any]],
                 **defaults):
        self.param_groups = Optimizer._group_params(params=params, defaults=defaults)

    @staticmethod
    def _group_params(params, defaults: Union[dict, None] = None) -> list[dict]:
        # Convert iterable of parameters into list of dictionaries
        param_groups = list(params)
        if all(isinstance(item, Parameter) for item in param_groups):
            param_groups = [{'params': param_groups}]

        # Convert single Parameters into a list of one Parameter and other iterables into lists
        for param_group in param_groups:
            params = param_group['params']
            if isinstance(params, Parameter):
                params = [params]
            else:
                params = list(params)
                if len(params) == 0:
                    raise ValueError("Empty parameter list passed to Optimizer")
            param_group['params'] = params

        # Add defaults if present
        if defaults is not None:
            for param_group in param_groups:
                keys_present = param_group.keys()
                param_group.update({key: val for key, val in defaults.items()
                                    if key not in keys_present})
        return param_groups

    @abstractmethod
    def step(self):
        """

        :return:
        """

    def zero_grad(self) -> None:
        for param_group in self.param_groups:
            for param in param_group['params']:
                param.zero_grad()
