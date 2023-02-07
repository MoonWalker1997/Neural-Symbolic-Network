import numpy as np


class IdentityLayer:
    """
        A layer mainly does one thing - it will make all objects forward and backward.
        But these objects are not only stored in an array, they might have a data-structure, e.g., matrix.
        So, a layer is to define the data-structure of inside objects.

        An identity layer is usually used for selection, say select 1 from 10.

        Since an identity layer is used in symbolic-level, it does not have a shape, or it will always be a 1D array.

        The backward function will only be in the identity layer, since the identity layer will always be the "output"
        layer in the current design.
    """

    def __init__(self, num_objects, ONodes):
        self.num_objects = num_objects
        self.objects = np.array([ONodes.produce("IDENTITY") for _ in range(num_objects)])

    def forward(self):
        for each_object in self.objects:
            each_object.value = each_object.forward()

    def backward(self, expected_values):
        for i, each_expected_value in enumerate(expected_values):
            if each_expected_value != self.objects[i].value:
                self.objects[i].backward(each_expected_value)
            else:
                self.objects[i].boost()
