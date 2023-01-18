import json

import numpy as np
from matplotlib import pyplot as plt

from InputLayer import InputLayer
from Layer import Layer


def connect(layer_1, layer_2):
    for i, each_object in enumerate(layer_2.input_SubLayer.objects):
        each_object.value = layer_1.output_SubLayer.objects[i].value
        each_object.parent_node = layer_1.output_SubLayer.objects[i]


class NSN:

    def __init__(self, ONodes):
        self.ONodes = ONodes
        self.input_layer = None
        self.internal_layers = []
        self.output_layer = None

    def forward(self, input_symbols):
        self.input_layer.forward(input_symbols)
        if len(self.internal_layers) != 0:
            connect(self.input_layer, self.internal_layers[0])
            for i in range(len(self.internal_layers) - 1):
                self.internal_layers[i].forward()
                connect(self.internal_layers[i], self.internal_layers[i + 1])
            self.internal_layers[-1].forward()
            connect(self.internal_layers[-1], self.output_layer)
        else:
            connect(self.input_layer, self.output_layer)
        self.output_layer.forward()

    def backward(self, expected_values):
        self.output_layer.backward(expected_values)

    def show_structure(self):
        print(self.input_layer.O_SubLayer.structure("input_layer"))
        for i in range(len(self.internal_layers)):
            print(self.internal_layers[i].random_O_SubLayer.structure("layer_" + str(i + 1)))
        print(self.output_layer.random_O_SubLayer.structure("output_layer"))
        print("--->")

    def save(self, path):
        # save the input layer
        IO, input_ONodes, input_ONodes_weight = self.input_layer.__save__()
        # save the internal layers
        ITO_internal = []
        internal_ONodes_1 = []
        internal_ONodes_weight_1 = []
        internal_ONodes_2 = []
        internal_ONodes_weight_2 = []
        for i in range(len(self.internal_layers)):
            ITO, internal_ONodes_1_i, internal_ONodes_weight_1_i, internal_ONodes_2_i, internal_ONodes_weight_2_i = \
                self.internal_layers[i].__save__()
            ITO_internal.append(ITO)
            internal_ONodes_1.append(internal_ONodes_1_i)
            internal_ONodes_weight_1.append(internal_ONodes_weight_1_i)
            internal_ONodes_2.append(internal_ONodes_2_i)
            internal_ONodes_weight_2.append(internal_ONodes_weight_2_i)
        # save the output layer
        ITO, output_ONodes_1, output_ONodes_weight_1, output_ONodes_2, output_ONodes_weight_2 = \
            self.output_layer.__save__()
        save_file = {"input": [IO, input_ONodes, input_ONodes_weight],
                     "internal": [ITO_internal, internal_ONodes_1, internal_ONodes_weight_1,
                                  internal_ONodes_2, internal_ONodes_weight_2],
                     "output": [ITO, output_ONodes_1, output_ONodes_weight_1, output_ONodes_2, output_ONodes_weight_2]}
        save_file_json = json.dumps(save_file)
        f = open(path, "w")
        f.write(save_file_json)

    def load(self, ONodes, load_file_path):
        load_file = open(load_file_path, "r")
        load_file = json.load(load_file)
        IO, input_ONodes, input_ONodes_weight = load_file["input"]
        ITO_internal, internal_ONodes_1, internal_ONodes_weight_1, internal_ONodes_2, internal_ONodes_weight_2 = \
            load_file["internal"]
        ITO, output_ONodes_1, output_ONodes_weight_1, output_ONodes_2, output_ONodes_weight_2 = load_file["output"]
        self.input_layer = InputLayer(IO[0], IO[1], ONodes, input_ONodes, input_ONodes_weight)
        for i in range(len(internal_ONodes_1)):
            self.internal_layers.append(Layer(ITO_internal[i][0],
                                              ITO_internal[i][1],
                                              ITO_internal[i][2], ONodes,
                                              internal_ONodes_1[i], internal_ONodes_weight_1[i],
                                              internal_ONodes_2[i], internal_ONodes_weight_2[i]))
        self.output_layer = Layer(ITO[0], ITO[1], ITO[2], ONodes,
                                  output_ONodes_1, output_ONodes_weight_1,
                                  output_ONodes_2, output_ONodes_weight_2)

    def analyze(self, show = False):
        matrix = [self.input_layer.analyze()]
        max_len = len(matrix[0])
        for each_layer in self.internal_layers:
            tmp = each_layer.analyze()
            matrix.append(tmp[0])
            max_len = max(max_len, len(tmp[0]))
            matrix.append(tmp[1])
            max_len = max(max_len, len(tmp[1]))
        tmp = self.output_layer.analyze()
        matrix.append(tmp[0])
        max_len = max(max_len, len(tmp[0]))
        matrix.append(tmp[1])
        max_len = max(max_len, len(tmp[1]))

        # matrix processing
        for i, each in enumerate(matrix):
            if len(each) < max_len:
                padding_left = (max_len - len(each)) // 2
                padding_right = max_len - len(each) - padding_left
                matrix[i] = [0.5 for _ in range(padding_left)] + matrix[i] + [0.5 for _ in range(padding_right)]

        if show:
            plt.imshow(matrix)
            plt.show()
        return matrix
