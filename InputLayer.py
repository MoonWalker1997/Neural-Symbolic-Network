from Nodes.InputNode import InputNode
from SubLayer import SubLayer


class InputLayer:
    """
    An input layer has 3 sub-layers, 2 S-sub-layers and 1 O-sub-layer. But input layer is different, it contains
    1 S-sub-layer and 1 O-sub-layer, and one additional "input-sub-layer".
    """

    def __init__(self, num_input, num_output, ONodes, load_file_nodes = None, load_file_weights = None):
        self.num_input = num_input
        self.num_output = num_output
        self.input_SubLayer = [InputNode() for _ in range(num_input)]
        self.output_SubLayer = SubLayer(num_output)  # 3rd, S-sub-layer
        if load_file_nodes is None:
            self.O_SubLayer = SubLayer(num_output, ONodes)  # 2nd, O-sub-layer
            self.connect()
        else:
            self.O_SubLayer = SubLayer(num_output, ONodes, load_file_nodes=load_file_nodes)
            self.connect(load_file_weights)

    def connect(self, load_file_weights = None):
        for i, each_object in enumerate(self.O_SubLayer.objects):
            if load_file_weights is None:
                each_object.input_weights = [0.5] + [0.5 / (self.num_input - 1) for _ in range(self.num_input - 1)]
            else:
                each_object.input_weights = load_file_weights[i]
            each_object.input_objects = self.input_SubLayer
            self.output_SubLayer.objects[i].parent_node = each_object

    def forward(self, input_symbols):
        # input values
        for i, each_object in enumerate(self.input_SubLayer):
            each_object.value = input_symbols[i]
        for i, each_object in enumerate(self.O_SubLayer.objects):
            # objects in the 1st S-sub-layer is known already, then calculate the value of each O-Node
            each_object.value = each_object.forward()
            # each O-Node is connected to one and only one S-Node, just pass it there
            self.output_SubLayer.objects[i].value = each_object.value

    def backward(self, expected_values):
        for i, each_expected_value in enumerate(expected_values):
            if each_expected_value != self.output_SubLayer.objects[i].value:
                self.output_SubLayer.objects[i].backward(each_expected_value)
            else:
                self.output_SubLayer.objects[i].boost()

    def show(self):
        print([each.value for each in self.output_SubLayer.objects])

    def __save__(self):
        IO = [self.num_input, self.num_output]
        ONodes = []
        ONodes_weight = []
        for each in self.O_SubLayer.objects:
            ONodes.append(each.name)
            ONodes_weight.append(each.input_weights)
        return IO, ONodes, ONodes_weight

    def analyze(self):
        """
        The input layer does not have an internal layer. So, its weight only varies in the O_SubLayer.
        """
        ret = []
        for each_object in self.O_SubLayer.objects:
            ret.append(each_object.analyze())
        return ret
