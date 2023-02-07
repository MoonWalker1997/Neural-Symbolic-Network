import numpy as np
import torch
from torchvision import datasets, transforms

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Resize([28, 28])])

data_train = datasets.MNIST(root="C:/Users/TORY/OneDrive - Temple University/AGI research/NARS_Optimizer/data/",
                            transform=transform,
                            train=True,
                            download=True)

data_test = datasets.MNIST(root="C:/Users/TORY/OneDrive - Temple University/AGI research/NARS_Optimizer/data/",
                           transform=transform,
                           train=False)

dataloader_train = torch.utils.data.DataLoader(dataset=data_train,
                                               batch_size=1,
                                               shuffle=True)

dataloader_test = torch.utils.data.DataLoader(dataset=data_test,
                                              batch_size=1,
                                              shuffle=True)


def MNIST_num(num, num_train, num_test, dataloader_train, dataloader_test):
    ret_train = []
    ret_test = []
    for x, y in dataloader_train:
        if len(ret_train) == num_train:
            break
        if num == y:
            x = np.reshape(x, (28 * 28))
            x = np.array([True if each > 0.5 else False for each in x]).reshape((1, 28, 28))
            ret_train.append(x)
    for x, y in dataloader_test:
        if len(ret_test) == num_test:
            break
        if num == y:
            x = np.reshape(x, (28 * 28))
            x = np.array([True if each > 0.5 else False for each in x]).reshape((1, 28, 28))
            ret_test.append(x)
    return ret_train, ret_test
