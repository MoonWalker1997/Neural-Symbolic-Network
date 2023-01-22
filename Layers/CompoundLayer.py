import random

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
