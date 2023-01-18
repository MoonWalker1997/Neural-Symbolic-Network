import random

from Nodes.AND_ONode import AND_ONode
from Nodes.ARROW_ONode import ARROW_ONode
from Nodes.IDENTITY_ONode import IDENTITY_ONode
from Nodes.SNode import SNode


class SubLayer:
    """
    There are three types of sub-layers, 1) S-sub-layer, 2) random O-sub-layer, 3) identity O-sub-layer.
    S-sub-layer only contains SNodes.
    Random O-sub-layer can have different numbers of different kinds of ONodes.
    Identity O-sub-layer can only identity ONodes.
    """

    def __init__(self, num_nodes, ONodes = None, identity = False, load_file_nodes = None):
        self.objects = []
        self.num_nodes = num_nodes
        if ONodes is not None:
            if not identity:
                self.type = "random O sub-layer"
                for i in range(num_nodes):
                    if load_file_nodes is None:
                        self.objects.append(ONodes.produce())
                    else:
                        self.objects.append(ONodes.produce(load_file_nodes[i]))
            else:
                self.type = "identity O sub-layer"
                for _ in range(num_nodes):
                    self.objects.append(IDENTITY_ONode())
        else:
            self.type = "S sub-layer"
            for _ in range(num_nodes):
                self.objects.append(SNode())

    def structure(self, name):
        num_AND = 0
        num_ARROW = 0
        num_IDENTITY = 0
        for each_object in self.objects:
            if isinstance(each_object, AND_ONode):
                num_AND += 1
            elif isinstance(each_object, ARROW_ONode):
                num_ARROW += 1
            else:
                num_IDENTITY += 1
        return name + " | AND:" + str(num_AND) + " | ARROW:" + str(num_ARROW) + " | IDENTITY:" + str(num_IDENTITY)
