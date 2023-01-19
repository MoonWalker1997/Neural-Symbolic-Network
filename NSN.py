import random

import networkx as nx
from matplotlib import pyplot as plt


def weights_initialization(num_objects):
    tmp = [0.5 / (num_objects - 1) for _ in range(num_objects)]
    tmp[random.randint(0, num_objects - 1)] = 0.5
    return tmp


def connect(layer_1, layer_2):
    # layer 2 is connected to layer 1
    for i, each_object in enumerate(layer_2.objects):
        each_object.input_weights = weights_initialization(layer_1.num_objects)
        each_object.input_objects = layer_1.objects


class NSN:

    def __init__(self, ONodes):
        self.ONodes = ONodes
        self.layers = []
        self.structure = []

    def connect(self):
        for i in range(len(self.layers) - 1):
            connect(self.layers[i], self.layers[i + 1])
        for each in self.layers:
            self.structure.append(each.num_objects)

    def forward(self, input_symbols):
        self.layers[0].forward(input_symbols)
        for i in range(1, len(self.layers)):
            self.layers[i].forward()

    def draw(self):
        weights = []
        for i in range(1, len(self.layers)):
            tmp = []
            for each_object in self.layers[i].objects:
                tmp.append(each_object.input_weights)
            tmp = zip(*tmp)
            for each in tmp:
                weights += each

        left, right, bottom, top, layer_sizes = .1, .9, .1, .9, self.structure

        G = nx.Graph()
        v_spacing = (top - bottom) / float(max(layer_sizes))
        h_spacing = (right - left) / float(len(layer_sizes) - 1)

        node_count = 0

        for i, v in enumerate(layer_sizes):
            layer_top = v_spacing * (v - 1) / 2. + (top + bottom) / 2.
            for j in range(v):
                G.add_node(node_count, pos=(left + i * h_spacing, layer_top - j * v_spacing))
                node_count += 1

        for x, (left_nodes, right_nodes) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            for i in range(left_nodes):
                for j in range(right_nodes):
                    G.add_edge(i + sum(layer_sizes[:x]), j + sum(layer_sizes[:x + 1]))

        pos = nx.get_node_attributes(G, 'pos')
        plt.figure()
        nx.draw(G, pos,
                node_color=range(node_count),
                with_labels=True,
                node_size=200,
                edge_color=weights,
                width=3,
                cmap=plt.cm.Dark2,
                edge_cmap=plt.cm.Blues
                )
        plt.show()
