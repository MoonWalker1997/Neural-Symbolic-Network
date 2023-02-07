import numpy as np

from Nodes.CNode import CNode


class CompoundLayer:
    """
        The old version of compound layer, which does not assume the spatial category, and all input symbols will be
        considered in making compounds, even in images.
    """

    def __init__(self, num_objects):
        self.num_objects = num_objects
        self.objects = [CNode() for _ in range(num_objects)]

    def forward(self):
        for each_object in self.objects:
            each_object.value = each_object.forward()

    def show(self, img_size):
        tmp = np.array([0.0 for _ in range(img_size[0] * img_size[1])])
        for each in self.objects:
            tmp += np.array(each.input_weights)
        tmp = np.reshape(tmp, img_size)
        return tmp
