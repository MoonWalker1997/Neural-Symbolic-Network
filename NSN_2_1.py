import numpy as np

from InputLayer import InputLayer
from Layer import Layer
from NSN import NSN


# def connect(layer_1, layer_2):
#     for i, each_object in enumerate(layer_2.input_SubLayer.objects):
#         each_object.value = layer_1.output_SubLayer.objects[i].value
#         each_object.parent_node = layer_1.output_SubLayer.objects[i]


class NSN_2(NSN):

    def __init__(self, ONodes, load_file_path=None):
        super(NSN_2, self).__init__(ONodes)
        if load_file_path is None:
            self.input_layer = InputLayer(10, 20, ONodes)
            # self.internal_layers.append(Layer(10, 40, 20, ONodes))
            self.output_layer = Layer(10, 20, 2, ONodes)
        else:
            self.load(ONodes, load_file_path)

    # def forward(self, input_symbols):
    #     self.input_layer.forward(input_symbols)
    #     if len(self.internal_layers) != 0:
    #         connect(self.input_layer, self.internal_layers[0])
    #         for i in range(len(self.internal_layers) - 1):
    #             self.internal_layers[i].forward()
    #             connect(self.internal_layers[i], self.internal_layers[i + 1])
    #         self.internal_layers[-1].forward()
    #         connect(self.internal_layers[-1], self.output_layer)
    #     else:
    #         connect(self.input_layer, self.output_layer)
    #     self.output_layer.forward()

    def backward(self, expected_values):
        """
        The first 5 are nested (OR operation), and the last 5 are also nested (OR operation), and we actually get 2
        outputs.
        """
        self.output_layer.backward(expected_values)

    # def show_structure(self):
    #     print(self.input_layer.O_SubLayer.structure("input_layer"))
    #     for i in range(len(self.internal_layers)):
    #         print(self.internal_layers[i].random_O_SubLayer.structure("layer_" + str(i + 1)))
    #     print(self.output_layer.random_O_SubLayer.structure("output_layer"))
    #     print("--->")
