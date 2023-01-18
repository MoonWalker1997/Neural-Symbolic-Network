from InputLayer import InputLayer
from Layer import Layer
from NSN import NSN


# def connect(layer_1, layer_2):
#     for i, each_object in enumerate(layer_2.input_SubLayer.objects):
#         each_object.value = layer_1.output_SubLayer.objects[i].value
#         each_object.parent_node = layer_1.output_SubLayer.objects[i]


class NSN_3(NSN):

    def __init__(self, ONodes, load_file_path = None):
        """
        It contains 2 layers, 1) one input layer and 2) one output layer.
        The size of the input layer is: 10 input symbols, 20 output symbols.
        The size of the output layer is: 20 input symbols, 20 internal nodes, and 2 output symbols.
        """
        super(NSN_3, self).__init__(ONodes)
        if load_file_path is None:
            self.input_layer = InputLayer(3, 3, ONodes)
            self.output_layer = Layer(3, 3, 1, ONodes)
        else:
            self.load(ONodes, load_file_path)

    def backward(self, expected_values):
        """
        The first 5 are nested (OR operation), and the last 5 are also nested (OR operation), and we actually get 2
        outputs.
        """
        self.output_layer.backward(expected_values)

    def structure(self):
        ret = []
        tmp = []
        for each in self.input_layer.O_SubLayer.objects:
            tmp.append(each.input_weights)
        ret.append(tmp)
        tmp = []
        for each in self.output_layer.random_O_SubLayer.objects:
            tmp.append(each.input_weights)
        ret.append(tmp)
        tmp = []
        for each in self.output_layer.identity_O_SubLayer.objects:
            tmp.append(each.input_weights)
        ret.append(tmp)
        print(ret)

