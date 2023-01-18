from SubLayer import SubLayer


class Layer:
    """
    A layer contains 4 sub-layers, 1 input layer, which is usually S-SubLayer, 1 output layer, which is an S-SubLayer.
    2 operation layers, one of them is a randomly generated operation layer, and its size is not related to the output.
    Another operation layer is fixed, which only contains "identity" nodes, which means a selection of previous
    intermediate results, and its size is the output size.
    """

    def __init__(self, num_input, num_internal, num_output, ONodes,
                 load_file_nodes_1 = None, load_file_weights_1 = None,
                 load_file_nodes_2 = None, load_file_weights_2 = None):
        self.num_input = num_input
        self.num_internal = num_internal
        self.num_output = num_output
        if load_file_nodes_1 is None:  # if this is None, all four are None
            self.input_SubLayer = SubLayer(num_input)  # 1st, S-sub-layer
            self.output_SubLayer = SubLayer(num_output)  # 4th, S-sub-layer
            self.random_O_SubLayer = SubLayer(num_internal, ONodes)  # 2nd, random O-sub-layer
            self.identity_O_SubLayer = SubLayer(num_output, ONodes, True)  # 3rd, identity O-sub-layer
            self.connect()
        else:
            self.input_SubLayer = SubLayer(num_input)
            self.output_SubLayer = SubLayer(num_output)
            self.random_O_SubLayer = SubLayer(num_internal, ONodes, load_file_nodes=load_file_nodes_1)
            self.identity_O_SubLayer = SubLayer(num_output, ONodes, load_file_nodes=load_file_nodes_2)
            self.connect(load_file_weights_1, load_file_weights_2)

    def connect(self, load_file_weights_1 = None, load_file_weights_2 = None):
        """
        Connect function makes connections between the first gap (between 1st and 2nd), the second gap
        (between 2nd and 3rd), and the third gap.
        """
        for i, each_object in enumerate(self.random_O_SubLayer.objects):
            if load_file_weights_1 is None:
                each_object.input_weights = [0.5] + [0.5 / (self.num_input - 1) for _ in range(self.num_input - 1)]
            else:
                each_object.input_weights = load_file_weights_1[i]
            each_object.input_objects = self.input_SubLayer.objects

        for i, each_object in enumerate(self.identity_O_SubLayer.objects):
            if load_file_weights_2 is None:
                each_object.input_weights = [0.5] + [0.5 / (self.num_internal - 1) for _ in
                                                     range(self.num_internal - 1)]
            else:
                each_object.input_weights = load_file_weights_2[i]
            each_object.input_objects = self.random_O_SubLayer.objects
            self.output_SubLayer.objects[i].parent_node = each_object

    def forward(self):
        for i, each_object in enumerate(self.random_O_SubLayer.objects):
            # objects in the 1st S-sub-layer is known already, then calculate the value of each random O-Node
            each_object.value = each_object.forward()

        for i, each_object in enumerate(self.identity_O_SubLayer.objects):
            # calculate the value of each identity O-Node
            each_object.value = each_object.forward()
            # each identity O-Node is connected to one and only one S-Node, just pass it there
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
        ITO = [self.num_input, self.num_internal, self.num_output]
        ONodes_1 = []
        ONodes_weight_1 = []
        ONodes_2 = []
        ONodes_weight_2 = []
        for each in self.random_O_SubLayer.objects:
            ONodes_1.append(each.name)
            ONodes_weight_1.append(each.input_weights)
        for each in self.identity_O_SubLayer.objects:
            ONodes_2.append(each.name)
            ONodes_weight_2.append(each.input_weights)
        return ITO, ONodes_1, ONodes_weight_1, ONodes_2, ONodes_weight_2

    def analyze(self):
        ret = [[], []]
        for each_object in self.random_O_SubLayer.objects:
            ret[0].append(each_object.analyze())
        for each_object in self.identity_O_SubLayer.objects:
            ret[1].append(each_object.analyze())
        return ret
