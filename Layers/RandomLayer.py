import numpy as np


class RandomLayer:
    """
        A layer mainly does one thing - it will make all objects forward and backward.
        But these objects are not only stored in an array, they might have a data-structure, e.g., matrix.
        So, a layer is to define the data-structure of inside objects.

        A random layer only contains O_Nodes. And there are no limitations of the types of these nodes. The reason why
        it is called "random" is that it just has a possibility for each type of ONode to appear, with no special spec.

        A random layer is also used in symbolic-level, it does not have a shape, or it will always be a 1D array.
    """

    def __init__(self, num_objects, ONodes):
        self.num_objects = num_objects
        self.objects = np.array([ONodes.produce() for _ in range(num_objects)])
        # the nodes_dictionary is used as a summary to show the number of each type of ONode.
        self.nodes_dictionary = ONodes.nodes_dictionary()
        for each in self.objects:
            self.nodes_dictionary[each.name] += 1

    def forward(self):
        for each_object in self.objects:
            each_object.value = each_object.forward()
