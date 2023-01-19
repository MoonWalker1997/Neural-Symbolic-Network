from Nodes.SNode import SNode


class InputLayer:
    """
    An input layer contains only S-Nodes.
    """

    def __init__(self, num_objects):
        self.num_objects = num_objects
        self.objects = [SNode() for _ in range(num_objects)]

    def forward(self, input_symbols):
        # input values
        for i, each_object in enumerate(self.objects):
            each_object.value = input_symbols[i]

    def backward(self, expected_values):
        # only change weights here
        pass

    def show(self):
        print([each.value for each in self.objects])
