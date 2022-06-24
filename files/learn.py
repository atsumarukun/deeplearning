import numpy as np
from tqdm import tqdm

from nets import *
from optmizers import *
from mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)
network = FoLCCAANet()
optmizer = AdaGrad()

for _ in tqdm(range(10000)):
    bach_mask = np.random.choice(x_train.shape[0], 100)
    x_bach = x_train[bach_mask]
    t_bach = t_train[bach_mask]
    
    grad = network.gradient(x_bach, t_bach)

    network.params = optmizer.update(network.params, grad)

print(network.accuracy(x_test, t_test))
