from Layers.CompoundLayer import CompoundLayer
from Layers.IdentityLayer import IdentityLayer
from Layers.InputLayer import InputLayer
from Layers.RandomLayer import RandomLayer
from NSN import NSN


class Propositional_Logic_NSN(NSN):

    def __init__(self, ONodes):
        super(Propositional_Logic_NSN, self).__init__(ONodes)
        self.first_layer_symbol = False
        self.input_layer = InputLayer(784)
        self.compound_layer_1 = CompoundLayer(5)
        self.compound_layer_2 = CompoundLayer(20)
        self.compound_layer_3 = CompoundLayer(20)
        self.hidden_layer_1 = RandomLayer(20, ONodes)
        self.hidden_layer_2 = RandomLayer(20, ONodes)
        self.hidden_layer_3 = RandomLayer(20, ONodes)
        self.output_layer = IdentityLayer(10, ONodes)
        for each_object in self.output_layer.objects:
            each_object.safeguard = 0.7
        self.layers = [self.input_layer,
                       self.compound_layer_1,
                       self.hidden_layer_1,
                       self.output_layer]
        self.connect()

    def backward(self, expected_values):
        self.output_layer.backward(expected_values)
