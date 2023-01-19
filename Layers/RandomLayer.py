


class RandomLayer:
    """
    A random layer only contains O_Nodes. And there are no limitations of the types of these nodes.
    """

    def __init__(self, num_objects, ONodes):
        self.num_objects = num_objects
        self.objects = [ONodes.produce() for _ in range(num_objects)]
        self.nodes_dictionary = ONodes.nodes_dictionary()
        for each in self.objects:
            self.nodes_dictionary[each.name] += 1

    def forward(self):
        for i, each_object in enumerate(self.objects):
            each_object.value = each_object.forward()

    def backward(self, expected_values):
        for i, each_expected_value in enumerate(expected_values):
            if each_expected_value != self.objects[i].value:
                self.objects[i].backward(each_expected_value)
            else:
                self.objects[i].boost()

    def show(self):
        print([each.value for each in self.output_SubLayer.objects])
