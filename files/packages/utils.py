import numpy as np
import gzip

def _load_img(file_path):
    with gzip.open(file_path, mode="rb") as f:
        imgs = np.frombuffer(f.read(), dtype=np.uint8, offset=16)
    return imgs.reshape(-1, 784)

def _load_label(file_path):
    with gzip.open(file_path, mode="rb") as f:
        labels = np.frombuffer(f.read(), dtype=np.uint8, offset=8)
    return labels

def _change_one_hot_label(labels):
    one_hot_label = np.zeros((labels.size, 10))
    for idx, row in enumerate(one_hot_label):
        row[labels[idx]] = 1
    return one_hot_label

def load_mnist(path, normalize=False, flatten=False, one_hot_label=False):
    dataset = {
            "train_img": _load_img(f"{path}/train-images-idx3-ubyte.gz"),
            "train_label": _load_label(f"{path}/train-labels-idx1-ubyte.gz"),
            "test_img": _load_img(f"{path}/t10k-images-idx3-ubyte.gz"),
            "test_label": _load_label(f"{path}/t10k-labels-idx1-ubyte.gz")
            }

    if normalize:
        for key in ("train_img", "test_img"):
            dataset[key] = dataset[key].astype(np.float32) / 255.0

    if not flatten:
        for key in ("train_img", "test_img"):
            dataset[key] = dataset[key].reshape(-1, 1, 28, 28)

    if one_hot_label:
        for key in ("train_label", "test_label"):
            dataset[key] = _change_one_hot_label(dataset[key])

    return (dataset["train_img"], dataset["train_label"]), (dataset["test_img"], dataset["test_label"])

def im2col(data, filter_size, stride=1, pad=0):
    N, C, H, W = data.shape
    out_h = (H + 2*pad - filter_size[0]) // stride + 1
    out_w = (W + 2*pad - filter_size[1]) // stride + 1

    img = np.pad(data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], "constant")
    col = np.zeros((N, C, filter_size[0], filter_size[1], out_h, out_w))

    for y in range(filter_size[0]):
        y_max = y + stride*out_h
        for x in range(filter_size[1]):
            x_max = x + stride*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

        return col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)

def col2im(data, input_size, filter_size, stride=1, pad=0):
    N, C, H, W = input_size
    out_h = (H + 2*pad - filter_size[0]) // stride + 1
    out_w = (W + 2*pad - filter_size[1]) // stride + 1

    col = data.reshape(N, out_h, out_w, C, filter_size[0], filter_size[1]).transpose(0, 3, 4, 5, 1, 2)
    img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))

    for y in range(filter_size[0]):
        y_max = y + stride*out_h
        for x in range(filter_size[1]):
            x_max = x + stride*out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]
