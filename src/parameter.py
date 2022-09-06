from . import Tensor
from .init import zeros


class Parameter(Tensor):
    """
    Does not override __array_function__ or __array_ufunc__, since we do not want operations on Parameters
    to produce new Parameters.

    TODO: Consider whether requires_grad should not always be True, we have Tensors
    for objects that do not have gradients.
    """

    def __new__(cls, data: Tensor, requires_grad: bool = True):
        obj = data.view(cls)

        # At this point __array_finalize__ has once been executed via the view method; thus,
        # if the object is instantiated via constructor, it already has _grad = None
        if requires_grad is True:
            obj._grad = zeros(obj.shape)
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self._grad = getattr(obj, '_grad', None)

    @property
    def grad(self) -> Tensor:
        return self._grad

    @grad.setter
    def grad(self, dx: Tensor):
        assert dx.shape == self.shape
        self._grad = dx

    def zero_grad(self):
        self.grad = zeros(self.shape)

    def optimize(self, dx: Tensor, *, maximize: bool = False):
        """
        Final optimization step deferred to the Parameter itself. Reason being that we might
        want to implement Parameter subclasses with their own constraints on the descent/ascent steps.
        """
        if maximize is True:
            self += dx
        else:
            self -= dx

    def __repr__(self):
        return f"{super().__repr__()} with gradient: {repr(self.grad)}"
