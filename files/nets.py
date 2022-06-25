import numpy as np
from layers import *

class BaseNet:
    def __init__(self):
        self.params = {}
        self.layers = {}

    def __predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        return self.lastLayer.forward(self.__predict(x), t)

    def accuracy(self, x, t):
        y = np.argmax(self.__predict(x), axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        return np.sum(y == t) / float(x.shape[0])

class TwLANet(BaseNet):
    def __init__(self, input_size=784, hidden_size=50, output_size=10, weight_init_std=0.01):
        super().__init__()

        self.params["W1"] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params["b1"] = np.zeros(hidden_size)
        self.params["W2"] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params["b2"] = np.zeros(output_size)

        self.layers["Affine1"] = Affine(self.params["W1"], self.params["b1"])
        self.layers["ReLU"] = ReLU()
        self.layers["Affine2"] = Affine(self.params["W2"], self.params["b2"])
        self.lastLayer = SoftmaxWithLoss()

    def gradient(self, x, t):
        self.loss(x, t)

        dx = self.lastLayer.backward(1)

        for layer in list(self.layers.values())[::-1]:
            dx = layer.backward(dx)

        grads = {}
        grads["W1"] = self.layers["Affine1"].dW
        grads["b1"] = self.layers["Affine1"].db
        grads["W2"] = self.layers["Affine2"].dW
        grads["b2"] = self.layers["Affine2"].db

        return grads

class ThLCAANet(BaseNet):
    def __init__(self, input_dim=(1, 28, 28), conv_params={"filter_num": 30, "filter_size": 5, "stride": 1, "pad": 0}, hidden_size=100, output_size=10, weight_init_std=0.01):
        super().__init__()

        filter_num = conv_params["filter_num"]
        filter_size = conv_params["filter_size"]
        filter_stride = conv_params["stride"]
        filter_pad = conv_params["pad"]
        input_size = input_dim[1]
        conv_output_size = (input_size - filter_size + 2*filter_pad) / filter_stride + 1
        pool_output_size = int(filter_num * (conv_output_size/2) ** 2)

        self.params["W1"] = weight_init_std * np.random.randn(filter_num, input_dim[0], filter_size, filter_size)
        self.params["b1"] = np.zeros(filter_num)
        self.params["W2"] = weight_init_std * np.random.randn(pool_output_size, hidden_size)
        self.params["b2"] = np.zeros(hidden_size)
        self.params["W3"] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params["b3"] = np.zeros(output_size)

        self.layers["Conv1"] = Convolution(self.params["W1"], self.params["b1"], conv_params["stride"], conv_params["pad"])
        self.layers["ReLU1"] = ReLU()
        self.layers["Pool1"] = Pooling((2, 2), 2, 0)
        self.layers["Dropout"] = Dropout(0.3)
        self.layers["Affine1"] = Affine(self.params["W2"], self.params["b2"])
        self.layers["ReLU2"] = ReLU()
        self.layers["Affine2"] = Affine(self.params["W3"], self.params["b3"])
        self.lastLayer = SoftmaxWithLoss()

    def gradient(self, x, t):
        self.loss(x, t)

        dx = self.lastLayer.backward(1)

        for layer in list(self.layers.values())[::-1]:
            dx = layer.backward(dx)

        grads = {}
        grads["W1"] = self.layers["Conv1"].dW
        grads["b1"] = self.layers["Conv1"].db
        grads["W2"] = self.layers["Affine1"].dW
        grads["b2"] = self.layers["Affine1"].db
        grads["W3"] = self.layers["Affine2"].dW
        grads["b3"] = self.layers["Affine2"].db

        return grads

class FoLCCAANet(BaseNet):
    def __init__(self, input_dim=(1, 28, 28), conv_params={"filter_num": 30, "filter_size": 5, "stride": 1, "pad": 0}, hidden_size=100, output_size=10, weight_init_std=0.01):
        super().__init__()

        filter_num = conv_params["filter_num"]
        filter_size = conv_params["filter_size"]
        filter_stride = conv_params["stride"]
        filter_pad = conv_params["pad"]
        input_size = input_dim[1]
        conv1_output_size = (input_size - filter_size + 2*filter_pad) / filter_stride + 1
        conv2_output_size = (conv1_output_size//2 - filter_size + 2*filter_pad) / filter_stride + 1
        pool_output_size = int(filter_num * (conv2_output_size/2) ** 2)

        self.params["W1"] = weight_init_std * np.random.randn(filter_num, input_dim[0], filter_size, filter_size)
        self.params["b1"] = np.zeros(filter_num)
        self.params["W2"] = weight_init_std + np.random.randn(filter_num, filter_num, filter_size, filter_size)
        self.params["b2"] = np.zeros(filter_num)
        self.params["W3"] = weight_init_std * np.random.randn(pool_output_size, hidden_size)
        self.params["b3"] = np.zeros(hidden_size)
        self.params["W4"] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params["b4"] = np.zeros(output_size)

        self.layers["Conv1"] = Convolution(self.params["W1"], self.params["b1"], conv_params["stride"], conv_params["pad"])
        self.layers["ReLU1"] = ReLU()
        self.layers["Pool1"] = Pooling((2, 2), 2, 0)
        self.layers["Conv2"] = Convolution(self.params["W2"], self.params["b2"])
        self.layers["ReLU2"] = ReLU()
        self.layers["Pool2"] = Pooling((2, 2), 2, 0)
        self.layers["Dropout"] = Dropout(0.25)
        self.layers["Affine1"] = Affine(self.params["W3"], self.params["b3"])
        self.layers["ReLU2"] = ReLU()
        self.layers["Affine2"] = Affine(self.params["W4"], self.params["b4"])
        self.lastLayer = SoftmaxWithLoss()

    def gradient(self, x, t):
        self.loss(x, t)

        dx = self.lastLayer.backward(1)

        for layer in list(self.layers.values())[::-1]:
            dx = layer.backward(dx)

        grads = {}
        grads["W1"] = self.layers["Conv1"].dW
        grads["b1"] = self.layers["Conv1"].db
        grads["W2"] = self.layers["Conv2"].dW
        grads["b2"] = self.layers["Conv2"].db
        grads["W3"] = self.layers["Affine1"].dW
        grads["b3"] = self.layers["Affine1"].db
        grads["W4"] = self.layers["Affine2"].dW
        grads["b4"] = self.layers["Affine2"].db

        return grads

class MyNet(BaseNet):
    def __init__(self):
        super().__init__()

        weight_init_std = 0.01
        hidden_size = 100
        output_size = 10

        self.params["W1"] = weight_init_std * np.random.randn(64, 1, 3, 3)
        conv1_out_size = self.__demand_conv_out_size(28, 3, 1, 1)
        self.params["b1"] = np.zeros(64)
        self.params["W2"] = weight_init_std * np.random.randn(64, 64, 3, 3)
        conv2_out_size = self.__demand_conv_out_size(conv1_out_size/2, 3, 1, 1)
        self.params["b2"] = np.zeros(64)
        self.params["W3"] = weight_init_std * np.random.randn(128, 64, 3, 3)
        conv3_out_size = self.__demand_conv_out_size(conv2_out_size/2, 3, 1, 1)
        self.params["b3"] = np.zeros(128)
        self.params["W4"] = weight_init_std * np.random.randn(128, 128, 3, 3)
        conv4_out_size = self.__demand_conv_out_size(conv3_out_size/2, 3, 1, 1)
        self.params["b4"] = np.zeros(128)
        affine_input_size = int(128 * int(conv4_out_size/2) ** 2)
        self.params["W5"] = weight_init_std * np.random.randn(affine_input_size, hidden_size)
        self.params["b5"] = np.zeros(hidden_size)
        self.params["W6"] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params["b6"] = np.zeros(output_size)

        self.layers["Conv1"] = Convolution(self.params["W1"], self.params["b1"], 1, 1)
        self.layers["ReLU1"] = ReLU()
        self.layers["Pool1"] = Pooling((2, 2), 2, 0)
        self.layers["Dropout1"] = Dropout(0.2)
        self.layers["Conv2"] = Convolution(self.params["W2"], self.params["b2"], 1, 1)
        self.layers["ReLU2"] = ReLU()
        self.layers["Pool2"] = Pooling((2, 2), 2, 0)
        self.layers["Dropout2"] = Dropout(0.2)
        self.layers["Conv3"] = Convolution(self.params["W3"], self.params["b3"], 1, 1)
        self.layers["ReLU3"] = ReLU()
        self.layers["Pool3"] = Pooling((2, 2), 2, 0)
        self.layers["Dropout3"] = Dropout(0.2)
        self.layers["Conv4"] = Convolution(self.params["W4"], self.params["b4"], 1, 1)
        self.layers["ReLU4"] = ReLU()
        self.layers["Pool4"] = Pooling((2, 2), 2, 0)
        self.layers["Dropout4"] = Dropout(0.2)
        self.layers["Affine1"] = Affine(self.params["W5"], self.params["b5"])
        self.layers["ReLU5"] = ReLU()
        self.layers["Dropout5"] = Dropout(0.5)
        self.layers["Affine2"] = Affine(self.params["W6"], self.params["b6"])
        self.lastLayer = SoftmaxWithLoss()

    def __demand_conv_out_size(self, input_size, filter_size, filter_pad, filter_stride):
        return int((input_size - filter_size + 2*filter_pad) / filter_stride + 1)

    def gradient(self, x, t):
        self.loss(x, t)

        dx = self.lastLayer.backward(1)

        for layer in list(self.layers.values())[::-1]:
            dx = layer.backward(dx)

        grads = {}
        grads["W1"] = self.layers["Conv1"].dW
        grads["b1"] = self.layers["Conv1"].db
        grads["W2"] = self.layers["Conv2"].dW
        grads["b2"] = self.layers["Conv2"].db
        grads["W3"] = self.layers["Conv3"].dW
        grads["b3"] = self.layers["Conv3"].db
        grads["W4"] = self.layers["Conv4"].dW
        grads["b4"] = self.layers["Conv4"].db
        grads["W5"] = self.layers["Affine1"].dW
        grads["b5"] = self.layers["Affine1"].db
        grads["W6"] = self.layers["Affine2"].dW
        grads["b6"] = self.layers["Affine2"].db

        return grads
