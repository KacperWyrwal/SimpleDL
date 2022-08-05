from nn import *


class MSELoss(Module):

    def __init__(self):
        self.sub = Sub()
        self.pow = Pow()
        self.mean = Mean()

    def forward(self, x, x_pred):
        y = self.sub(x, x_pred)
        y = self.pow(y, 2)
        y = self.mean(y)
        return y

    def backward(self, lr):
        dy = self.mean.backward(lr)
        dy = self.pow.backward(dy)
        dx, dx_pred = self.sub.backward(dy)
        return dx_pred
