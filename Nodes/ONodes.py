import numpy as np
from Nodes.AND_ONode import AND_ONode
from Nodes.ARROW_ONode import ARROW_ONode
from Nodes.IDENTITY_ONode import IDENTITY_ONode


class ONodes:
    """
    A class for managing all O-Nodes. New O-Nodes might be added, by modifying this class, these changes can be applied.
    """

    def __init__(self):
        self.num_type = 3

    def produce(self, name = None):
        """
        Currently, only the number of nodes is given, and different O-Nodes will be added randomly, following a uniform
        distribution.
        """
        if name is None:
            name = np.random.choice(["AND", "ARROW", "IDENTITY"], p=[1 / self.num_type for _ in range(self.num_type)])
        if name == "AND":
            return AND_ONode()
        elif name == "ARROW":
            return ARROW_ONode()
        else:
            return IDENTITY_ONode()

    def nodes_dictionary(self):
        return {"AND Operation": 0,
                "ARROW Operation": 0,
                "IDENTITY Operation": 0}
