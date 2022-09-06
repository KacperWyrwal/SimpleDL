from __future__ import annotations

from abc import ABC, abstractmethod
from .. import Parameter

from typing import Any, Tuple, Iterator


class Module(ABC):

    def __init__(self):
        self._parameters = None
        self._modules = None

    @abstractmethod
    def forward(self, *args, **kwargs) -> Any:
        """
        :param args:
        :param kwargs:
        :return:
        """

    @abstractmethod
    def backward(self, *args, **kwargs) -> Any:
        """
        :param args:
        :param kwargs:
        :return:
        """

    def __call__(self, *args, **kwargs) -> Any:
        return self.forward(*args, **kwargs)

    def named_modules(self, recurse: bool = True) -> Iterator[Tuple[str, Module]]:
        """
        Generates submodules and their names of this Module object in the order in which they appear
        in the __dict__ attribute. If recurse is True then subsubmodules of submodules etc. until reaching
        submodules with no subsubmodules.
        """
        if self._modules is None:
            self._register_modules_from_attributes()
        for name, module in self._modules.items():
            yield name, module
            if recurse is True:
                for subname, submodule in module.named_modules():
                    yield f"{name}.{subname}", submodule

    def modules(self, recurse: bool = True) -> Iterator[Module]:
        for _, module in self.named_modules(recurse=recurse):
            yield module

    def named_parameters(self, recurse: bool = True) -> Iterator[Tuple[str, Parameter]]:
        if self._parameters is None:
            self._register_parameters_from_attributes()
        for name, param in self._parameters.items():
            yield name, param
        if recurse is True:
            for module_name, module in self.named_modules(recurse=False):
                for param_name, param in module.named_parameters(recurse=True):
                    yield f"{module_name}.{param_name}", param

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        for _, parameter in self.named_parameters(recurse=recurse):
            yield parameter

    def _register_modules_from_attributes(self) -> None:
        self._modules = {attr: val for attr, val in self.__dict__.items()
                         if issubclass(type(val), Module)}

    def _register_parameters_from_attributes(self) -> None:
        self._parameters = {attr: val for attr, val in self.__dict__.items()
                            if issubclass(type(val), Parameter)}
