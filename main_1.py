import tqdm

import pickle

import numpy as np

from NSN_1 import NSN_1
from Nodes.ONodes import ONodes

ONodes_1 = ONodes()
NSN_1 = NSN_1(ONodes_1)


def unpickle(file):
    with open(file, "rb") as fo:
        d = pickle.load(fo, encoding="bytes")
    return d


batch1 = unpickle(
    "C:/Users/TORY/OneDrive - Temple University/Fall 2022/CIS 5543/Assignment 3/cifar-10-batches-py/data_batch_1")
batch2 = unpickle(
    "C:/Users/TORY/OneDrive - Temple University/Fall 2022/CIS 5543/Assignment 3/cifar-10-batches-py/data_batch_2")
batch3 = unpickle(
    "C:/Users/TORY/OneDrive - Temple University/Fall 2022/CIS 5543/Assignment 3/cifar-10-batches-py/data_batch_3")
batch4 = unpickle(
    "C:/Users/TORY/OneDrive - Temple University/Fall 2022/CIS 5543/Assignment 3/cifar-10-batches-py/data_batch_4")
batch5 = unpickle(
    "C:/Users/TORY/OneDrive - Temple University/Fall 2022/CIS 5543/Assignment 3/cifar-10-batches-py/data_batch_5")
test_batch = unpickle(
    "C:/Users/TORY/OneDrive - Temple University/Fall 2022/CIS 5543/Assignment 3/cifar-10-batches-py/test_batch")
X = np.vstack([batch1[b'data'], batch2[b'data'], batch3[b'data'], batch4[b'data'], batch5[b'data']])
Y = np.array(batch1[b'labels'] + batch2[b'labels'] + batch3[b'labels'] + batch4[b'labels'] + batch5[b'labels'])
validation = np.random.choice(range(len(X)), 1000, replace=False)
train = np.setdiff1d(range(len(X)), validation)
X_validation = X[validation]
Y_validation = Y[validation]
X_train = X[train]
Y_train = Y[train]
training_cap = 500
testing_cap = 50
validation_cap = 50
epoch = 10


def util_X(cifar_input):
    ret = [
        True if (cifar_input[i] * 299 + cifar_input[i + 1024] * 587 + cifar_input[i + 2048] * 114) > 125000 else False
        for i in range(1024)]
    return ret


def util_Y(cifar_label):
    ret = [False for _ in range(10)]
    ret[cifar_label] = True
    return ret


if __name__ == '__main__':
    for _ in tqdm.tqdm(range(epoch)):
        # training process
        for i in tqdm.tqdm(range(len(X_train))):
            if i < training_cap:
                x = util_X(X_train[i])
                y = util_Y(Y_train[i])
                NSN_1.forward(x)
                NSN_1.backward(y)
        # validating process
        succeed = 0
        fail = 0
        for i in tqdm.tqdm(range(len(X_validation))):
            if i < validation_cap:
                x = util_X(X_train[i])
                y = util_Y(Y_train[i])
                NSN_1.forward(x)
                if y == all([each.value for each in NSN_1.output_layer.output_SubLayer.objects]):
                    succeed += 1
                else:
                    fail += 1

        print(succeed / (succeed + fail))
    # NSN_1.forward([True, True, True, True, True, True, True, True, True, True])
    # NSN_1.backward([True, False, True, True, False, True, True, False, True, True])
