import numpy as np
from numpy.lib.stride_tricks import as_strided

from abc import ABC, abstractmethod
from init import normal
from utils import unpad


class Module(ABC):
    """
    Superclass of all Neural Network modules.
    """

    @abstractmethod
    def forward(self, *args, **kwargs):
        ...

    @abstractmethod
    def backward(self, *args, **kwargs):
        ...

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Linear(Module):

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        self.in_features = in_features
        self.out_features = out_features

        # Might want to make this additional dimension abstract
        self.weights = normal((out_features, in_features))
        self.bias = normal((out_features, 1)) if bias else None

        self.add = Add()
        self.mul = MatMul()

    def forward(self, x):
        y = self.mul(self.weights, x)
        y = self.add(self.bias, y)
        return y

    def backward(self, dy, lr=0.1):
        db, dy = self.add.backward(dy)
        dw, dx = self.mul.backward(dy)

        self.bias -= db * lr
        self.weights -= dw * lr
        return dx


class MSELoss(Module):

    def __init__(self):
        self.x = None
        self.x_hat = None

    def forward(self, x, x_hat):
        self.x = x
        self.x_hat = x_hat
        return np.mean((x - x_hat) ** 2, keepdims=True)

    def backward(self, dy=1):
        return 2 * (self.x_hat - self.x) / len(self.x) * dy


# TODO Implement more arithmetic import Sigmoid and ReLU


class ReLU(Module):

    def __init__(self):
        self.x = None

    def forward(self, x):
        self.x = x
        return x * (0 < x)

    def backward(self, dy):
        return (0 < self.x) * dy


class Sigmoid(Module):

    def __init__(self):
        self.x = None

    def forward(self, x):
        self.x = x
        return np.exp(x) / (1 + np.exp(x))

    def backward(self, dy):
        return (np.exp(self.x) / (1 + np.exp(self.x)) ** 2).T * dy


"""
Arithmetic
"""

import numpy as np

from nn import Module


class Add(Module):

    def __init__(self):
        self.x1 = None
        self.x2 = None

    def forward(self, x1, x2):
        self.x1 = x1
        self.x2 = x2
        return x1 + x2

    def backward(self, dy):
        return dy, dy


class Sub(Module):

    def __init__(self):
        self.x1 = None
        self.x2 = None

    def forward(self, x1, x2):
        self.x1 = x1
        self.x2 = x2
        return x1 - x2

    def backward(self, dy):
        return dy, -dy


class MatMul(Module):

    def __init__(self):
        self.x1 = None
        self.x2 = None

    def forward(self, x1, x2):
        self.x1 = x1
        self.x2 = x2
        return self.x1 @ self.x2

    def backward(self, dy):
        dx1 = dy @ self.x2.T
        dx2 = self.x1.T @ dy

        return dx1, dx2


class Pow(Module):

    def __init__(self):
        self.x = None
        self.p = None

    def forward(self, x, p):
        self.x = x
        self.p = p
        return np.power(x, p)

    def backward(self, dy):
        return (self.p - 1) * np.power(self.x, self.p - 1)


class Mean(Module):

    def __init__(self):
        self.x = None

    def forward(self, x):
        self.x = x
        return np.mean(x)

    def backward(self, dy):
        return np.broadcast_to(dy / len(self.x), self.x.shape)


class Mul(Module):

    def __init__(self):
        self.x1 = None
        self.x2 = None

    def forward(self, x1, x2):
        self.x1 = x1
        self.x2 = x2
        return x1 * x2

    def backward(self, dy):
        return self.x2 * dy, self.x1 * dy


class Div(Mul):

    def forward(self, x1, x2):
        return super().forward(x1, 1 / x2)


class Exp(Module):

    def __init__(self):
        self.x = None

    def forward(self, x):
        self.x = x
        return np.exp(x)

    def backward(self, dy):
        return dy * np.exp(self.x)


class AsStrided(Module):
    """
    For some reason breaks when input was created using np.arange(...).reshape(...), but works in all other
    cases tested.
    """

    def __init__(self, shape, strides, *, writeable: bool = False):
        self.shape = shape
        self.strides = strides
        self.writeable = writeable
        self.x = None

    def forward(self, x):
        self.x = x
        return as_strided(x, self.shape, self.strides, writeable=self.writeable)

    def backward(self, dy):
        dx = np.zeros(self.x.shape)
        dx_strided = as_strided(dx, self.shape, self.strides, writeable=True)

        assert dx_strided.shape == dy.shape
        for i in np.ndindex(dy.shape):
            dx_strided[i] += dy[i]
        return dx


class Sum(Module):

    def __init__(self, axis=None):
        self.axis = axis
        self.x = None

    def forward(self, x):
        self.x = x
        return np.sum(x, axis=self.axis)

    def backward(self, dy):
        # This could probably be done in a more straight-forward manner
        strides = np.ones(len(self.x.strides), dtype=int)
        strides.put(self.axis, 0)
        strides[strides != 0] = dy.strides
        return np.lib.stride_tricks.as_strided(
            dy,
            self.x.shape,
            tuple(strides),
            writeable=False,
        )


class Windows(Module):
    
    def __init__(self, kernel_shape, *, stride=1, padding=0, dilation=1):
        # Parameters
        self.kernel_shape = kernel_shape
        self.stride = stride
        self.padding = padding
        self.dilation = dilation 
        
        # Modules 
        self.pad = Pad(self.padding)
        self.as_strided = None 
        
    def _outer_shape(self, x_shape):
        # adjust for kernel shape and dilation 
        outer_shape = np.subtract(x_shape[:-1], np.subtract(self.kernel_shape, 1) * self.dilation)
        
        # adjust for stride 
        return np.ceil(np.divide(outer_shape, self.stride)).astype(int)
    
    def _outer_strides(self, x_strides):
        return np.multiply(x_strides[:-1], self.stride)
    
    def _kernel_strides(self, x_strides):
        return np.multiply(x_strides[:-1], self.dilation)
    
    @abstractmethod 
    def _shape(self, x_shape):
        """
        Returns the shape of the window view. Utilize _outer_shape and kernel_shape.
        """
        
    @abstractmethod
    def _strides(self, x_strides):
        """
        Returns the strides of the window view. Utilize _outer_strides and _kernel_strides
        """
    
    def forward(self, x):
        x = self.pad(x)
        self.as_strided = AsStrided(self._shape(x.shape), self._strides(x.strides))
        return self.as_strided(x)
    
    def backward(self, dy):
        dy = self.as_strided.backward(dy)
        return self.pad.backward(dy)
    

class PoolWindows(Windows):
    
    def __init__(self, channels, kernel_shape, *, stride=1, padding=0, dilation=1):
        super().__init__(kernel_shape=kernel_shape, stride=stride, padding=padding, dilation=dilation)
        self.channels = channels 
    
    def _shape(self, x_shape):
        return *self._outer_shape(x_shape=x_shape), self.channels, *self.kernel_shape
    
    def _strides(self, x_strides):
        return *self._outer_strides(x_strides=x_strides), x_strides[-1], \
    *self._kernel_strides(x_strides=x_strides)

    
class ConvWindows(Windows):
    
    def __init__(self, in_channels, out_channels, kernel_shape, *, stride=1, padding=0, dilation=1):
        super().__init__(kernel_shape=kernel_shape, stride=stride, padding=padding, dilation=dilation)
        self.in_channels = in_channels
        self.out_channels = out_channels 
        
    def _shape(self, x_shape):
        return *self._outer_shape(x_shape=x_shape), self.out_channels, *self.kernel_shape, self.in_channels
    
    def _strides(self, x_strides):
        return *self._outer_strides(x_strides=x_strides), 0, *self._kernel_strides(x_strides=x_strides), \
    x_strides[-1]


class Conv(Module):

    def __init__(self, in_channels, out_channels, kernel_shape, *, stride=1,
                 padding=0, dilation=1, bias=True):
        # Parameters
        self.weights = normal((out_channels, *kernel_shape, in_channels))
        self.bias = normal((out_channels, 1)) if bias is True else None
        self.ndim = len(kernel_shape)

        # Modules
        self.input_to_windows = ConvWindows(out_channels=out_channels, kernel_shape=kernel_shape,
                                            in_channels=in_channels, stride=stride, padding=padding,
                                            dilation=dilation)
        self.weights_to_windows = None
        self.bias_to_windows = None
        self.mul = Mul()
        self.sum = Sum(axis=tuple(range(-self.ndim-1, 0)))
        self.add = Add() if bias is True else None

    def forward(self, x):
        x = self.input_to_windows(x)

        self.weights_to_windows = AsStrided(x.shape, (*(0,)*self.ndim, *self.weights.strides))
        k = self.weights_to_windows(self.weights)

        y = self.mul(x, k)
        y = self.sum(y)

        if self.bias is not None:
            self.bias_to_windows = AsStrided(y.shape, (*(0,)*self.ndim, self.bias.strides[-1]))
            b = self.bias_to_windows(self.bias)
            y = self.add(y, b)
        return y

    def backward(self, dy, lr=0.1):
        if self.bias is not None:
            dy, dy_b = self.add.backward(dy)

            db = self.bias_to_windows.backward(dy_b)
            self.bias -= db * lr

        dy = self.sum.backward(dy)
        dy_x, dy_w = self.mul.backward(dy)

        dw = self.weights_to_windows.backward(dy_w)
        self.weights -= dw * lr

        dx = self.input_to_windows.backward(dy_x)
        return dx


class Reshape(Module):

    def __init__(self, *shape):
        self.shape = shape
        self.x = None

    def forward(self, x):
        self.x = x
        return self.x.reshape(self.shape)

    def backward(self, dy):
        return dy.reshape(self.x.shape)


class Flatten(Reshape):

    def __init__(self):
        super().__init__(-1, 1)


class Pad(Module):

    def __init__(self, padding):
        self.padding = padding

    def forward(self, x):
        return np.pad(x, self.padding)

    def backward(self, dy):
        return unpad(dy, self.padding)
