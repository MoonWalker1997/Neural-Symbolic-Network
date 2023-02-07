import numpy as np

from Nodes.IDENTITY_ONode import IDENTITY_ONode


class CompoundPoolLayer:
    """
        A layer mainly does one thing - it will make all objects forward and backward.
        But these objects are not only stored in an array, they might have a data-structure, e.g., matrix.
        So, a layer is to define the data-structure of inside objects.

        A compound pooling layer is used for compound pooling, say you have a 10x10 spatial compounds, by pooling,
        you can have 5x5. Most likely the pooling process in image processing.

        Currently, only 2x2 pooling is supported, and it is by default.

        Different from these symbolic-level layers, it is in feature-level, so its input is bounded by some categories,
        here, specifically the spatial category. And so, its objects have a spatial-bounded data-structure. This is
        achieved by self.set_objects()
    """

    def __init__(self, shape):
        self.objects = []
        self.set_objects(shape)

    def set_objects(self, shape):
        """
            Currently, this layer is only used for image processing, and so its input must have the shape (c, i, j).

            This pooling will be the pooling for these "feature maps", so it will not change the number of channels.
        """
        for c in range(shape[0]):
            tmp1 = []
            for i in range(shape[1] // 2):
                tmp2 = []
                for j in range(shape[2] // 2):
                    tmp_idt = IDENTITY_ONode()
                    tmp2.append(tmp_idt)
                tmp1.append(tmp2)
            self.objects.append(tmp1)
        self.objects = np.array(self.objects)

    def forward(self):
        for c in range(self.objects.shape[0]):  # channels
            for i in range(self.objects.shape[1]):  # columns
                for j in range(self.objects.shape[2]):  # rows
                    self.objects[c, i, j].value = self.objects[c, i, j].forward()
