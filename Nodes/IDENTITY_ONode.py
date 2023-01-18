import numpy as np

from Nodes.InputNode import InputNode
from Nodes.ONode import ONode


class IDENTITY_ONode(ONode):
    """
    Except for AND and ARROW, since the topological structure of NN is fixed, each layer can only be derived from the
    direct previous layer, so this identity function is used to expand the representation power.
    """

    def __init__(self):
        super(IDENTITY_ONode, self).__init__()
        self.name = "IDENTITY Operation"
        self.index = -1
        self.value_decay = 0.5
        self.weight_decay = 0.3
        self.award = 1.5

    def forward(self):
        self.index = np.random.choice(range(len(self.input_objects)), p=self.input_weights)
        return self.input_objects[self.index].value

    def backward(self, expected_value):
        """
        By calling this function, this means there must be some mistakes to correct.
        """
        if isinstance(self.input_objects[self.index], InputNode):  # If this mistake happens in the input layer.
            # We cannot push the mistake further back, therefore, we can only change the weights, but values.
            self.input_weights[self.index] *= self.weight_decay
            # print("weight changed")
        else:
            approach = np.random.choice(["value", "weight"],
                                        p=[self.input_weights[self.index], 1 - self.input_weights[self.index]])
            if approach == "value":
                self.input_objects[self.index].backward(expected_value)
                self.input_weights[self.index] *= self.value_decay
                # print("value changed")
            else:
                self.input_weights[self.index] *= self.weight_decay
                # print("weight changed")
        # weight regularize
        self.weight_regularize()

    def boost(self):
        self.input_weights[self.index] *= self.award
        self.weight_regularize()
