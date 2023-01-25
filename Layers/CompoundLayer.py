import random

import matplotlib.pyplot as plt
import numpy as np

from Nodes.CNode import CNode


class CompoundLayer:
    """
    Currently, there are hierarchies for compounds are considered.

    A compound layer will process all inputs in this layer and provide several compounds.
    """

    def __init__(self, num_objects):
        self.num_objects = num_objects
        self.objects = [CNode() for _ in range(num_objects)]

    def forward(self):
        for each_object in self.objects:
            each_object.value = each_object.forward()

    def backward(self, expected_values):
        for i, each_expected_value in enumerate(expected_values):
            if each_expected_value != self.objects[i].value:
                self.objects[i].backward(each_expected_value)
            else:
                self.objects[i].boost()

    def show(self, img_size):
        tmp = np.array([0.0 for _ in range(img_size[0] * img_size[1])])
        for each in self.objects:
            tmp += np.array(each.input_weights)
        tmp = np.reshape(tmp, img_size)
        return tmp
