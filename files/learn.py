import numpy as np
from tqdm import tqdm
import pickle5

from nets import *
from optmizers import *
from mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)
params={"filter_num": 64, "filter_size": 3, "stride": 1, "pad": 2}
network = ThLCAANet(conv_params=params)
optmizer = Adam()
# network = pickle5.load(open("/files/fashion_mnist.pkl", "rb"))

for i in tqdm(range(10000)):
    bach_mask = np.random.choice(x_train.shape[0], 100)
    x_bach = x_train[bach_mask]
    t_bach = t_train[bach_mask]
    
    grad = network.gradient(x_bach, t_bach)

    network.params = optmizer.update(network.params, grad)

    if not i % 1000:
        print(network.accuracy(x_train[:1000], t_train[:1000]))
        print(network.accuracy(x_test, t_test))
pickle5.dump(network, open("/files/fashion_mnist.pkl", "wb"))

print(network.accuracy(x_test, t_test))
