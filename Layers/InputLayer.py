import numpy as np

from Nodes.SNode import SNode


class InputLayer:
    """
        A layer mainly does one thing - it will make all objects forward and backward.
        But these objects are not only stored in an array, they might have a data-structure, e.g., matrix.
        So, a layer is to define the data-structure of inside objects.

        An input layer is only used an interface for inputting. It only has SNodes (not backward).

        Currently, the input layer is only used for image inputting, and so its shape will always be (c, i, j).
    """

    def __init__(self, shape):
        self.shape = shape  # input shape, also the output shape, this shape must have form (c, i, j)
        self.objects = np.array(
            [[[SNode() for _ in range(shape[2])] for _ in range(shape[1])] for _ in range(shape[0])])

    def forward(self, x):
        # input values
        for c in range(self.shape[0]):
            for i in range(self.shape[1]):
                for j in range(self.shape[2]):
                    self.objects[c, i, j].value = x[c, i, j]
