from Layers.CompoundConvLayer import CompoundConvLayer
from Layers.CompoundLayer import CompoundLayer
from Layers.CompoundPoolLayer import CompoundPoolLayer
from Layers.IdentityLayer import IdentityLayer
from Layers.InputLayer import InputLayer
from Layers.RandomLayer import RandomLayer
from NSN import NSN


class Convolution_NSN(NSN):

    def __init__(self, ONodes):
        super(Convolution_NSN, self).__init__(ONodes)
        self.input_layer = InputLayer((1, 28, 28))  # all input nodes are SNodes

        self.compound_layer_1 = CompoundConvLayer(5, (1, 28, 28))
        # 5 3x3 "convolutions" on a 1x28x28 matrix, no padding, stride 1
        # after it, the size will be 5x26x26

        self.compound_layer_2 = CompoundConvLayer(3, (5, 26, 26))
        # 5 3x3 "convolutions" on a 5x26x26 matrix, no padding, stride 1
        # after it, the size will be 3x24x24

        self.compound_layer_3 = CompoundPoolLayer((3, 24, 24))
        # A 2x2 "pooling" on a 3x24x24 matrix, no padding, stride 1
        # after it, the size will be 3x12x12

        self.compound_layer_4 = CompoundConvLayer(1, (3, 12, 12))
        # 5 3x3 "convolutions" on a 3x12x12 matrix, no padding, stride 1
        # after it, the size will be 1x10x10

        self.compound_layer_5 = CompoundPoolLayer((1, 10, 10))
        # A 2x2 "pooling" on a 5x10x10 matrix, no padding, stride 1
        # after it, the size will be 1x5x5, and the total number of symbols will be 25, and sparse enough for
        # symbolic reasoning

        self.symbolic_layer_1 = RandomLayer(25, ONodes)
        self.symbolic_layer_2 = RandomLayer(20, ONodes)
        self.output_layer = IdentityLayer(10, ONodes)
        for each_object in self.output_layer.objects:
            each_object.safeguard = 0.7
        self.layers = [self.input_layer,
                       self.compound_layer_1,
                       self.compound_layer_2,
                       self.compound_layer_3,
                       self.compound_layer_4,
                       self.compound_layer_5,
                       self.symbolic_layer_1,
                       self.symbolic_layer_2,
                       self.output_layer]
        self.connect()

    def backward(self, expected_values):
        self.output_layer.backward(expected_values)
