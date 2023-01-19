import networkx as nx
from Layers.IdentityLayer import IdentityLayer
from Layers.InputLayer import InputLayer
from Layers.RandomLayer import RandomLayer
from matplotlib import pyplot as plt
from NSN import NSN



class Propositional_Logic_NSN(NSN):

    def __init__(self, ONodes):
        super(Propositional_Logic_NSN, self).__init__(ONodes)
        self.input_layer = InputLayer(10)
        self.hidden_layer_1 = RandomLayer(5, ONodes)
        self.hidden_layer_2 = RandomLayer(5, ONodes)
        self.output_layer = IdentityLayer(1, ONodes)
        self.layers = [self.input_layer, self.hidden_layer_1, self.hidden_layer_2, self.output_layer]
        self.connect()

    def backward(self, expected_values):
        self.output_layer.backward(expected_values)
