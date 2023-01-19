from Nodes.Node import Node


class SNode(Node):
    """
    This class is for "symbol nodes", which means the value of these nodes are "symbols".
    """

    def __init__(self):
        super(SNode, self).__init__()
        self.value = None  # e.g., string, number, ...
        self.parent_node = None

    def boost(self):
        self.parent_node.boost()
