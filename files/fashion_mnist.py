import numpy as np
import pickle5
import sys
from tqdm import tqdm

from packages.nets import *
from packages.optmizers import *
from packages.utils import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist("dataset/fashion_mnist/", normalize=True, one_hot_label=True)
optmizer = Adam()

def learn():
    params = {"filter_num": 32, "filter_size":3, "stride": 1, "pad": 2}
    network = FashionMnistNet(conv_params=params, hidden_size=200)
    # network = TwLANet(hidden_size=200)

    for i in tqdm(range(10000)):
        bach_mask = np.random.choice(x_train.shape[0], 100)
        x_bach = x_train[bach_mask]
        t_bach = t_train[bach_mask]

        grad = network.gradient(x_bach, t_bach)
        network.params = optmizer.update(network.params, grad)

        if not i % 1000:
            print(f"[{i // 1000}] Train accuracy: {network.accuracy(x_train[:1000], t_train[:1000])}, Test accuracy: {network.accuracy(x_test, t_test)}")

    print(f"Test accuracy: {network.accuracy(x_test, t_test)}")
    pickle5.dump(network, open("/files/params/fashion_mnist.pkl", "wb"))

def verification():
    with open("/files/params/fashion_mnist.pkl", "rb") as f:
        network = pickle5.load(f)
    print(f"Test accuracy: {network.accuracy(x_test, t_test)}")

def main():
    try:
        eval(f"{sys.argv[1]}()")
    except NameError:
        print("No such argument.")
    except KeyboardInterrupt:
        print("Interrupts learning.")
    except IndexError:
        learn()

if __name__ == "__main__":
    main()
