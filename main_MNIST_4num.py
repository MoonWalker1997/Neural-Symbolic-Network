import numpy as np
import torch
from matplotlib import pyplot as plt

from Nodes.ONodes import ONodes
from models.MNIST_NSN_784_X_X_10 import Propositional_Logic_NSN
from util.MNIST_data import data_train, data_test, MNIST_num

ONodes = ONodes()
NSN = Propositional_Logic_NSN(ONodes)

if __name__ == '__main__':

    """
    The training process is seperated into several stages. Each stage will have different contents to learn.
    And the same stage might be repeated for reviewing.

    1) Training for #8, 5 epochs.
    2) Training for #3, 5 epochs.
    3) Training for #0, 5 epochs.
    4) Training for #7, 5 epochs.
    
    """

    great_loop = 500
    for _ in range(great_loop):

        plt.figure()
        img_idx = 1

        # 1st Stage
        # ==============================================================================================================
        # spec
        print("===================")
        num = 8
        num_epoch = 5
        num_train = 10
        num_test = 200
        # data generation
        dataloader_train = torch.utils.data.DataLoader(dataset=data_train, batch_size=1, shuffle=True)
        dataloader_test = torch.utils.data.DataLoader(dataset=data_test, batch_size=1, shuffle=True)
        train, test = MNIST_num(num, num_train, num_test, dataloader_train, dataloader_test)
        # cycle
        for _ in range(num_epoch):
            # training
            for x in train:
                NSN.forward(x)
                tmp = [False for _ in range(10)]
                tmp[num] = True
                NSN.backward(tmp)
            plt.subplot(4, num_epoch, img_idx)
            img_idx += 1
            plt.imshow(NSN.compound_layer_1.show((28, 28)), cmap="Reds")
            # time.sleep(3)

            # testing
            succeed = 0
            fail = 0
            for x in test:
                NSN.forward(x)
                if num == np.argmax(np.array([each.value for each in NSN.output_layer.objects])):
                    succeed += 1
                else:
                    fail += 1
            print("1st stage, Acc:", succeed / (succeed + fail))
        # ==============================================================================================================

        # 2nd Stage
        # ==================================================================================================================
        # spec
        print("===================")
        num = 3
        num_epoch = 5
        num_train = 10
        num_test = 200
        # data generation
        dataloader_train = torch.utils.data.DataLoader(dataset=data_train, batch_size=1, shuffle=True)
        dataloader_test = torch.utils.data.DataLoader(dataset=data_test, batch_size=1, shuffle=True)
        train, test = MNIST_num(num, num_train, num_test, dataloader_train, dataloader_test)
        # cycle
        for _ in range(num_epoch):
            # training
            for x in train:
                NSN.forward(x)
                tmp = [False for _ in range(10)]
                tmp[num] = True
                NSN.backward(tmp)
            plt.subplot(4, num_epoch, img_idx)
            img_idx += 1
            plt.imshow(NSN.compound_layer_1.show((28, 28)), cmap="Blues")
            # time.sleep(3)

            # testing
            succeed = 0
            fail = 0
            for x in test:
                NSN.forward(x)
                if num == np.argmax(np.array([each.value for each in NSN.output_layer.objects])):
                    succeed += 1
                else:
                    fail += 1
            print("2nd stage, Acc:", succeed / (succeed + fail))
        # ==============================================================================================================

        # 3rd Stage
        # ==============================================================================================================
        # spec
        print("===================")
        num = 0
        num_epoch = 5
        num_train = 10
        num_test = 200
        # data generation
        dataloader_train = torch.utils.data.DataLoader(dataset=data_train, batch_size=1, shuffle=True)
        dataloader_test = torch.utils.data.DataLoader(dataset=data_test, batch_size=1, shuffle=True)
        train, test = MNIST_num(num, num_train, num_test, dataloader_train, dataloader_test)
        # cycle
        for _ in range(num_epoch):
            # training
            for x in train:
                NSN.forward(x)
                tmp = [False for _ in range(10)]
                tmp[num] = True
                NSN.backward(tmp)
            plt.subplot(4, num_epoch, img_idx)
            img_idx += 1
            plt.imshow(NSN.compound_layer_1.show((28, 28)), cmap="Greens")
            # time.sleep(3)

            # testing
            succeed = 0
            fail = 0
            for x in test:
                NSN.forward(x)
                if num == np.argmax(np.array([each.value for each in NSN.output_layer.objects])):
                    succeed += 1
                else:
                    fail += 1
            print("3rd stage, Acc:", succeed / (succeed + fail))
        # ==============================================================================================================

        # 4th Stage
        # ==============================================================================================================
        # spec
        print("===================")
        num = 7
        num_epoch = 5
        num_train = 10
        num_test = 200
        # data generation
        dataloader_train = torch.utils.data.DataLoader(dataset=data_train, batch_size=1, shuffle=True)
        dataloader_test = torch.utils.data.DataLoader(dataset=data_test, batch_size=1, shuffle=True)
        train, test = MNIST_num(num, num_train, num_test, dataloader_train, dataloader_test)
        # cycle
        for _ in range(num_epoch):
            # training
            for x in train:
                NSN.forward(x)
                tmp = [False for _ in range(10)]
                tmp[num] = True
                NSN.backward(tmp)
            plt.subplot(4, num_epoch, img_idx)
            img_idx += 1
            plt.imshow(NSN.compound_layer_1.show((28, 28)), cmap="Purples")
            # time.sleep(3)

            # testing
            succeed = 0
            fail = 0
            for x in test:
                NSN.forward(x)
                if num == np.argmax(np.array([each.value for each in NSN.output_layer.objects])):
                    succeed += 1
                else:
                    fail += 1
            print("4th stage, Acc:", succeed / (succeed + fail))
        # ==============================================================================================================

        plt.show()
