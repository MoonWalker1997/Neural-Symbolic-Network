from InputLayer import InputLayer
from Layer import Layer


def connect(layer_1, layer_2):
    for i, each_object in enumerate(layer_2.input_SubLayer.objects):
        each_object.value = layer_1.output_SubLayer.objects[i].value
        each_object.parent_node = layer_1.output_SubLayer.objects[i]


class NSN_1:

    def __init__(self, ONodes):
        self.input_layer = InputLayer(1024, 1024, ONodes)
        self.layer_1 = Layer(1024, 1024, ONodes)
        self.layer_2 = Layer(1024, 1024, ONodes)
        self.layer_3 = Layer(1024, 1024, ONodes)
        self.layer_4 = Layer(1024, 1024, ONodes)
        self.layer_5 = Layer(1024, 1024, ONodes)
        self.output_layer = Layer(1024, 10, ONodes)

    def forward(self, input_symbols):
        self.input_layer.forward(input_symbols)
        connect(self.input_layer, self.layer_1)
        self.layer_1.forward()
        connect(self.layer_1, self.layer_2)
        self.layer_2.forward()
        connect(self.layer_2, self.layer_3)
        self.layer_3.forward()
        connect(self.layer_3, self.layer_4)
        self.layer_4.forward()
        connect(self.layer_4, self.layer_5)
        self.layer_5.forward()
        connect(self.layer_5, self.output_layer)
        self.output_layer.forward()

    def backward(self, expected_values):
        self.output_layer.backward(expected_values)
