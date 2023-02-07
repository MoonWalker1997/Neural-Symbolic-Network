import numpy as np
import torch
from matplotlib import pyplot as plt

from Nodes.ONodes import ONodes
from models.MNIST_NSN_conv import Convolution_NSN
from util.MNIST_data import data_train, data_test, MNIST_num

ONodes = ONodes()
NSN = Convolution_NSN(ONodes)

if __name__ == '__main__':

    """
    The training process is seperated into several stages. Each stage will have different contents to learn.
    And the same stage might be repeated for reviewing.

    1) Training for #8, 5 epochs.
    2) Training for #3, 5 epochs.
    3) repeat, ...
    """

    great_loop = 20
    type_1_learning_curve = []
    type_2_learning_curve = []

    for gl in range(great_loop):

        if 5 < gl <= 10:
            for each_object in NSN.compound_layer_1.objects:
                each_object.mark = False
        elif 10 < gl <= 15:
            for each_object in NSN.compound_layer_1.objects:
                each_object.mark = True
        elif gl > 15:
            for each_object in NSN.compound_layer_1.objects:
                each_object.mark = False

        plt.figure()
        img_idx = 1

        # 1st Stage
        # ==============================================================================================================
        # spec
        print("===================")
        num = 8
        num_epoch = 5
        num_train = 30
        num_test = 200
        # data generation
        dataloader_train = torch.utils.data.DataLoader(dataset=data_train, batch_size=1, shuffle=True)
        dataloader_test = torch.utils.data.DataLoader(dataset=data_test, batch_size=1, shuffle=True)
        train, test = MNIST_num(num, num_train, num_test, dataloader_train, dataloader_test)
        # cycle
        tmp_curve = []
        for _ in range(num_epoch):
            # training
            for x in train:
                NSN.forward(x)
                tmp = [False for _ in range(10)]
                tmp[num] = True
                NSN.backward(tmp)
            # plt.subplot(2, num_epoch, img_idx)
            # img_idx += 1
            # plt.imshow(NSN.compound_layer_1.show((28, 28)), cmap="Reds")
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
            tmp_curve.append(succeed / (succeed + fail))
            print("1st stage, Acc:", tmp_curve[-1])
        type_1_learning_curve.append(tmp_curve)
        # ==============================================================================================================

        # for each_object in NSN.compound_layer_1.objects:
        #     print(len(each_object.indices))

        # 2nd Stage
        # ==============================================================================================================
        # spec
        print("===================")
        num = 3
        num_epoch = 5
        num_train = 30
        num_test = 200
        # data generation
        dataloader_train = torch.utils.data.DataLoader(dataset=data_train, batch_size=1, shuffle=True)
        dataloader_test = torch.utils.data.DataLoader(dataset=data_test, batch_size=1, shuffle=True)
        train, test = MNIST_num(num, num_train, num_test, dataloader_train, dataloader_test)
        # cycle
        tmp_curve = []
        for _ in range(num_epoch):
            # training
            for x in train:
                NSN.forward(x)
                tmp = [False for _ in range(10)]
                tmp[num] = True
                NSN.backward(tmp)
            # plt.subplot(2, num_epoch, img_idx)
            # img_idx += 1
            # plt.imshow(NSN.compound_layer_1.show((28, 28)), cmap="Blues")
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
            tmp_curve.append(succeed / (succeed + fail))
            print("2nd stage, Acc:", tmp_curve[-1])
        type_2_learning_curve.append(tmp_curve)
        # ==============================================================================================================

        # for each_object in NSN.compound_layer_1.objects:
        #     print(len(each_object.indices))

    #     plt.show()
    #     NSN.draw(starting_layer=6)
    #
    plt.figure()
    for each in type_1_learning_curve:
        plt.plot(each)
    plt.show()

    plt.figure()
    for each in type_2_learning_curve:
        plt.plot(each)
    plt.show()
