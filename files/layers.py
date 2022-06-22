import numpy as np

from functions import *
from utils import *

class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.x_shape = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x_shape = x.shape
        self.x = x.reshape(x.shape[0], -1)
        return np.dot(self.x, self.W) + self.b

    def backward(self, dx):
        self.dW = np.dot(self.x.T, dx)
        self.db = np.sum(dx, axis=0)
        return np.dot(dx, self.W.T).reshape(*self.x_shape)

class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad
        self.x = None
        self.col = None
        self.col_W = None
        self.dW = None
        self.db = None

    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = int(1 + (H + 2*self.pad - FH) / self.stride)
        out_w = int(1 + (W + 2*self.pad - FW) / self.stride)

        col = im2col(x, (FH, FW), self.stride, self.pad)
        col_W = self.W.reshape(FN, -1).T
        out = np.dot(col, col_W) + self.b

        self.x = x
        self.col = col
        self.col_W = col_W
        return out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

    def backward(self, dx):
        FN, C, FH, FW = self.W.shape
        dx = dx.transpose(0, 2, 3, 1).reshape(-1, FN)

        self.dW = np.dot(self.col.T, dx).transpose(1, 0).reshape(FN, C, FH, FW)
        self.db = np.sum(dx, axis=0)
        return col2im(np.dot(dx, self.col_W.T), self.x.shape, (FH, FW), self.stride, self.pad)

class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        self.out = sigmoid(x)
        return self.out

    def backward(self, dx):
        return dx * (1.0 - self.out) * self.out

class ReLU:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        x[self.mask] = 0
        return x

    def backward(self, dx):
        dx[self.mask] = 0
        return dx

class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = CELoss(self.y, self.t)
        return self.loss

    def backward(self, dx=1):
        if self.t.size == self.y.size:
            return (self.y - self.t) / self.t.shape[0]
        else:
            dx = self.y.copy()
            dx[np.arange(self.t.shape[0]), self.t] -= 1
            return dx / self.t.shape[0]

class Pooling:
    def __init__(self, pool_size, stride=2, pad=0):
        self.pool_size = pool_size
        self.stride = stride
        self.pad = pad
        self.x = None
        self.arg_max = None

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_size[0]) / self.stride)
        out_w = int(1 + (W - self.pool_size[1]) / self.stride)

        col = im2col(x, self.pool_size, self.stride, self.pad).reshape(-1, self.pool_size[0]*self.pool_size[1])

        self.x = x
        self.arg_max = np.argmax(col, axis=1)
        return np.max(col, axis=1).reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

    def backward(self, dx):
        dx = dx.transpose(0, 2, 3, 1)

        pool_size = self.pool_size[0]*self.pool_size[1]
        dmax = np.zeros((dx.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dx.flatten()
        dmax = dmax.reshape(dx.shape + (pool_size, ))

        return col2im(dmax.reshape(dmax.shape[0]*dmax.shape[1]*dmax.shape[2], -1), self.x.shape, self.pool_size, self.stride, self.pad)
