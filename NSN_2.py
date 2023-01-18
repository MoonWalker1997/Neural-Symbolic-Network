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
        value_0 = self.output_layer.output_SubLayer.objects[0].value or \
                  self.output_layer.output_SubLayer.objects[1].value or \
                  self.output_layer.output_SubLayer.objects[2].value or \
                  self.output_layer.output_SubLayer.objects[3].value or \
                  self.output_layer.output_SubLayer.objects[4].value
        value_1 = self.output_layer.output_SubLayer.objects[5].value or \
                  self.output_layer.output_SubLayer.objects[6].value or \
                  self.output_layer.output_SubLayer.objects[7].value or \
                  self.output_layer.output_SubLayer.objects[8].value or \
                  self.output_layer.output_SubLayer.objects[9].value
        if value_0 != expected_values[0]:
            if expected_values[0]:
                # TODO: this might not be like this, I should pick the one most worthy of changing
                expected_values_t = [True, False, False, False, False]
            else:
                expected_values_t = [False, False, False, False, False]
        else:
            expected_values_t = [self.output_layer.output_SubLayer.objects[0].value,
                                 self.output_layer.output_SubLayer.objects[1].value,
                                 self.output_layer.output_SubLayer.objects[2].value,
                                 self.output_layer.output_SubLayer.objects[3].value,
                                 self.output_layer.output_SubLayer.objects[4].value]
        if value_1 != expected_values[1]:
            if expected_values[0]:
                # TODO: this might not be like this, I should pick the one most worthy of changing
                expected_values_t += [True, False, False, False, False]
            else:
                expected_values_t += [False, False, False, False, False]
        else:
            expected_values_t += [self.output_layer.output_SubLayer.objects[5].value,
                                  self.output_layer.output_SubLayer.objects[6].value,
                                  self.output_layer.output_SubLayer.objects[7].value,
                                  self.output_layer.output_SubLayer.objects[8].value,
                                  self.output_layer.output_SubLayer.objects[9].value]

        self.output_layer.backward(expected_values_t)

    # def show_structure(self):
    #     print(self.input_layer.O_SubLayer.structure("input_layer"))
    #     for i in range(len(self.internal_layers)):
    #         print(self.internal_layers[i].random_O_SubLayer.structure("layer_" + str(i + 1)))
    #     print(self.output_layer.random_O_SubLayer.structure("output_layer"))
    #     print("--->")
