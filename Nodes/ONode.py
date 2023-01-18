import numpy as np

from Nodes.Node import Node
from abc import abstractmethod


class ONode(Node):
    """
    This class is for "operation nodes", which means the value of these nodes are "symbols".
    """

    def __init__(self):
        # operation nodes have no values, but they may have different names
        # e.g., "AND operation node", "OR operation node"
        super(ONode, self).__init__()

    @abstractmethod
    def forward(self):
        """
        Consider self.input_weights, get the output value.
        """
        pass

    @abstractmethod
    def backward(self, **kwargs):
        """
        Weights update, punishing.
        """
        pass

    @abstractmethod
    def boost(self):
        """
        Weights update, encouraging.
        """
        pass

    def weight_regularize(self):
        # 1st method, just regularize
        self.input_weights = list(np.array(self.input_weights) / sum(self.input_weights))
        # 2nd method, softmax
        # tmp = [np.exp(each) for each in self.input_weights]
        # sum_tmp = sum(tmp)
        # self.input_weights = [each/sum_tmp for each in tmp]

    def analyze(self):
        if max(self.input_weights) / (self.input_weights[np.argsort(self.input_weights)[-2]] + 1e-5) >= 1.1:
            return 1
        else:
            return 0
