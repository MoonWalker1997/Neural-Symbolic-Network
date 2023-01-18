import numpy as np

from Nodes.InputNode import InputNode
from Nodes.ONode import ONode

"""
A sample operation node in propositional logic NSN.
"""


class ARROW_ONode(ONode):

    def __init__(self):
        super(ARROW_ONode, self).__init__()
        self.name = "ARROW Operation"
        self.index_1 = -1
        self.index_2 = -1
        self.value_decay = 0.5
        self.weight_decay = 0.3
        self.award = 1.5

    def forward(self):
        # find the indices of involved objects
        [self.index_1, self.index_2] = np.random.choice(a=range(len(self.input_objects)),
                                                        size=2,
                                                        replace=False,
                                                        p=self.input_weights)
        return (not self.input_objects[self.index_1].value) or self.input_objects[self.index_2].value

    def backward(self, expected_value):
        # find the index of the weight to update
        # the smaller the weight, the higher tha chance to be updated
        p = np.array([1 - self.input_weights[self.index_1], 1 - self.input_weights[self.index_2]])
        to_update = np.random.choice([self.index_1, self.index_2], p=p / p.sum())
        # figure out the way to update
        approach = np.random.choice(["value", "weight"],
                                    p=[self.input_weights[to_update], 1 - self.input_weights[to_update]])
        # keep the weight (choice) while change the value
        if approach == "value":
            if isinstance(self.input_objects[to_update], InputNode):
                approach = "weight"
            else:
                value_remain = self.input_objects[self.index_1 if to_update == self.index_2 else self.index_2].value
                if expected_value:
                    if to_update == self.index_1:
                        self.input_objects[to_update].backward(False)
                        self.input_weights[to_update] *= self.value_decay
                        # print("value changed")
                    else:
                        self.input_objects[to_update].backward(True)
                        self.input_weights[to_update] *= self.value_decay
                        # print("value changed")
                else:
                    if value_remain:
                        if to_update == self.index_1:
                            self.input_objects[to_update].backward(False)
                            self.input_weights[to_update] *= self.value_decay
                            # print("value changed")
                        else:
                            approach = "weight"
                    else:
                        if to_update == self.index_1:
                            approach = "weight"
                        else:
                            self.input_objects[to_update].backward(True)
                            self.input_weights[to_update] *= self.value_decay
                            # print("value changed")
        # Pick another weight
        if approach == "weight":
            self.input_weights[to_update] *= self.weight_decay
            # print("weight changed")
        # weight regularize
        self.weight_regularize()

    def boost(self):
        self.input_weights[self.index_1] *= self.award
        self.input_weights[self.index_2] *= self.award
        self.weight_regularize()
