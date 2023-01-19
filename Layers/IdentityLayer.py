


class IdentityLayer:
    """
    The identity layer is used for changing the size.
    For example, you have 100 internal symbols, but your label is only 10 digit.
    Then you can use 10 identity O_Node to do so.
    """

    def __init__(self, num_objects, ONodes):
        self.num_objects = num_objects
        self.objects = [ONodes.produce("IDENTITY") for _ in range(num_objects)]

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
