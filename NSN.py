import random

import networkx as nx
from matplotlib import pyplot as plt

from Layers import CompoundConvLayer, CompoundPoolLayer


def weights_initialization(num_objects, cpd_layer = False):
    if not cpd_layer:
        if num_objects != 1:
            tmp = [0.5 / (num_objects - 1) for _ in range(num_objects)]
            tmp[random.randint(0, num_objects - 1)] = 0.5
            return tmp
        else:
            return [1]
    else:
        return [random.random() for _ in range(num_objects)]


def connect(layer_1, layer_2):
    """
        The way of connecting two layers is decided by the type of the layer that is connected with, say layer_2.

        And currently, this connection is limited, even with a fixed structure, say:
        InputLayer -> CompoundConvLayer/CompoundPoolLayer(*) -> RandomLayer(*) -> IdentityLayer
    """
    if isinstance(layer_2, CompoundConvLayer.CompoundConvLayer):
        # in this case, layer_1 can either be an InputLayer or a CompoundConv/PoolLayer
        # in all cases, the shape of the input objects must be (c, i, j)
        if isinstance(layer_1, CompoundConvLayer.CompoundConvLayer):
            for i in range(layer_2.objects.shape[1]):
                for j in range(layer_2.objects.shape[2]):
                    for c in range(layer_2.objects.shape[0]):
                        layer_2.objects[c, i, j].input_objects = list(
                            layer_1.idt_objects[:, i:i + 3, j:j + 3].flatten())
                        layer_2.objects[c, i, j].input_weights = weights_initialization(
                            layer_1.idt_objects.shape[0] * 9, cpd_layer=True)
                        layer_2.objects[c, i, j].pattern = [random.random() for _ in
                                                            range(layer_1.idt_objects.shape[0] * 9)]
        else:
            for i in range(layer_2.objects.shape[1]):
                for j in range(layer_2.objects.shape[2]):
                    for c in range(layer_2.objects.shape[0]):
                        layer_2.objects[c, i, j].input_objects = list(layer_1.objects[:, i:i + 3, j:j + 3].flatten())
                        layer_2.objects[c, i, j].input_weights = weights_initialization(layer_1.objects.shape[0] * 9,
                                                                                        cpd_layer=True)
                        layer_2.objects[c, i, j].pattern = [random.random() for _ in
                                                            range(layer_1.objects.shape[0] * 9)]
    elif isinstance(layer_2, CompoundPoolLayer.CompoundPoolLayer):
        # in this case, layer_1 can only be a CompoundConvLayer and the shape of the input objects must be (c, i, j)
        # and the "output" objects in layer_1 should be "idt_objects"
        for i in range(layer_2.objects.shape[1]):
            for j in range(layer_2.objects.shape[2]):
                for c in range(layer_2.objects.shape[0]):
                    try:
                        layer_2.objects[c, i, j].input_objects = list(
                            layer_1.idt_objects[:, 2 * i:2 * i + 2, 2 * j:2 * j + 2].flatten())
                    except:
                        print(1)
                    layer_2.objects[c, i, j].input_weights = weights_initialization(layer_1.idt_objects.shape[0] * 4)
                    layer_2.objects[c, i, j].pattern = [random.random() for _ in
                                                        range(layer_1.idt_objects.shape[0] * 4)]
    else:
        # in this case, layer_2 must be one symbolic-level layer, but layer_1 can be a CompoundConv/PoolLayer or
        # another symbolic-level layer
        # but by all means, all layer_1 nodes are connected to each node in layer_2
        if isinstance(layer_1, CompoundConvLayer.CompoundConvLayer):
            for each in layer_2.objects:
                each.input_objects = layer_1.idt_objects.flatten()
                each.input_weights = weights_initialization(each.input_objects.shape[0])
        else:
            for each in layer_2.objects:
                each.input_objects = layer_1.objects.flatten()
                each.input_weights = weights_initialization(each.input_objects.shape[0])


class NSN:

    def __init__(self, ONodes):
        self.ONodes = ONodes
        self.layers = []
        self.structure = []

    def connect(self):
        for i in range(len(self.layers) - 1):
            connect(self.layers[i], self.layers[i + 1])
        for each in self.layers:
            if not isinstance(each, CompoundConvLayer.CompoundConvLayer) and \
                    not isinstance(each, CompoundPoolLayer.CompoundPoolLayer):
                self.structure.append(each.objects.shape[0])

    def forward(self, input_symbols):
        self.layers[0].forward(input_symbols)
        for i in range(1, len(self.layers)):
            self.layers[i].forward()

    def draw(self, starting_layer):
        weights = []
        for i in range(starting_layer, len(self.layers)):
            tmp = []
            for each_object in self.layers[i].objects:
                tmp.append(each_object.input_weights)
            tmp = zip(*tmp)
            for each in tmp:
                weights += each

        left, right, bottom, top, layer_sizes = .1, .9, .1, .9, self.structure[starting_layer:]

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
                width=1,
                cmap=plt.cm.Dark2,
                edge_cmap=plt.cm.Blues
                )
        plt.show()
