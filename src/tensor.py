import numpy as np

from numpy.typing import ArrayLike
from typing import Sequence


class Tensor(np.ndarray):

    def __new__(cls, data: ArrayLike):
        obj = np.asarray(data).view(cls)
        return obj

    def __array_ufunc__(self, ufunc, method, *inputs, out=None, **kwargs):
        # input validation
        Tensor._validate_inputs(inputs)
        Tensor._validate_inputs(kwargs.values())
        if out is not None:
            Tensor._validate_inputs(out)

            # The subclasses of Tensor must be downcasted to np.arrays
            out = Tensor._downcast_where_possible(out)
        args = Tensor._downcast_where_possible(inputs)
        kwargs = Tensor._downcast_where_possible(kwargs)
        results = super().__array_ufunc__(ufunc, method, *args, out=out, **kwargs)

        # return output Tensors if possible
        if out is not None:
            if ufunc.nout == 1:
                results = (results,)
            results = tuple(result if output is None else output
                            for result, output in zip(results, out))
            if len(results) == 1:
                results = results[0]

        # Upcast to Tensor
        return Tensor._upcast_where_possible(results)

    def __array_function__(self, func, types, *args, **kwargs):
        """
        Somehow array_function works without the downcasting, yet also without visiting __array_ufunc__
        """
        Tensor._validate_input_types(types)
        results = super().__array_function__(func, types, *args, **kwargs)
        return Tensor._upcast_where_possible(results)

    @staticmethod
    def _downcast_where_possible(iterable):
        if isinstance(iterable, dict):
            return dict(zip(iterable.keys(), Tensor._downcast_where_possible(iterable.values())))
        return tuple(item.view(np.ndarray) if issubclass(type(item), np.ndarray) else item
                     for item in iterable)

    @staticmethod
    def _upcast_where_possible(results):
        """
        If the result is an np.ndarray subclass, return a Tensor. If it is an iterable, return it with all
        subclasses of np.ndarray casted to Tensor.
        """
        # np.array subclass
        if issubclass(type(results), np.ndarray):
            return results.view(Tensor)
        # Iterable
        try:
            return type(results)(
                result.view(Tensor) if issubclass(type(result), np.ndarray) else result
                for result in results
            )
        # single, non-iterable object - turn to Tensor
        # TODO: add proper exception
        except:
            return Tensor(results)

    @staticmethod
    def _validate_inputs(inputs: Sequence):
        types = map(type, inputs)
        Tensor._validate_input_types(types)

    @staticmethod
    def _validate_input_types(types):
        """
        Does not accept np.ndarray superclasses
        """
        if any(issubclass(Tensor, type_) and Tensor is not type_ for type_ in types):
            raise NotImplemented
