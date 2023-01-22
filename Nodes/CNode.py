import random

import numpy as np

from Nodes.Node import Node
from Nodes.SNode import SNode


class CNode(Node):
    """
    CNode is for "compound node".

    It is necessary to address the difference of all three types of nodes (O, S, CNodes):
    1) ONode, operations node, it contains one truth function, which will give an output based on the input according
       to the truth function. Once a mistake is made, we know how to make corrections. And this is hard-coded.
    2) SNode, the simplest form of symbols, which contains no operations, and so no truth functions.
    3) CNode, which will give a compound upon inputs. For example, in an 100*100 image, we just pick some pixels to
       get a compound. It will also give an "output", but it is not based on a "fixed" truth function.
       Since whether a compound is true, is based on whether the input is consistent with the pattern. But the pattern
       is not fixed.
    """

    def __init__(self):
        super(CNode, self).__init__()
        """
        Specially, in CNodes, sum(self.input_weights) is not one anymore.
        Each input weight is independent, and range from 0 to 1.
        """
        self.pattern = []
        self.indices = []  # as a compound may have many components, it has many indices
        self.score = 0
        self.threshold = 0.5
        self.weight_decay = 0.9
        self.pattern_lr = 0.1
        self.award = 1.1

    def forward(self):
        for i in range(len(self.input_weights)):
            # TODO, is it possible to get a "weighted average" ?
            if random.random() < self.input_weights[i]:  # based on the weights (attention), this position is selected
                self.indices.append(i)
                self.score += abs(self.pattern[i] - self.input_objects[i].value)
        if self.score != 0:
            self.score /= len(self.indices)
        if self.score > self.threshold:
            return True
        else:
            return False

    def backward(self, expected_value):
        """
        Different from  ONodes, which has two "failure types":
        1) weight error (my attention is not good)
        2) value error (my attention is good, but what I am focusing is incorrect)

        CNodes has 3 failure types:
        1) weight error (my attention is not good)
        2) pattern error (my attention is good, but my pattern is not good)
        3) value error (mu attention and pattern is good, but something is wrong with the values forwarded)

        Specially, in CNodes, there is a possibility with no punishment when making mistakes.
        * When the weight is really high, and when the pattern is really confident, and my pattern is consistent with
        the input, then this position will not be updated.
        """
        for each in range(len(self.indices)):
            if random.random() < self.input_weights[each]:
                # my attention is good
                if random.random() * 0.5 < abs(self.pattern[each] - 0.5):
                    # my pattern is good
                    if (self.pattern[each] > 0.5 and self.input_weights[each]) or (
                            self.pattern[each] <= 0.5 and not self.input_weights[each]):
                        # my pattern is consistent with the input then no problem
                        pass
                    else:
                        # my pattern is inconsistent with the input
                        # something is wrong with the values
                        self.input_weights[each] *= self.weight_decay
                        if not isinstance(self.input_objects[each], SNode):
                            # there is a parent compound layer
                            if self.pattern[each] > 0.5:
                                # my pattern thinks this should be true
                                self.input_objects[each].backward(True)
                            else:
                                self.input_objects[each].backward(False)
                        else:
                            # otherwise, I have no choice to think something is wrong with this pattern
                            if expected_value:
                                # if I want to make my result "True", I need to make my pattern close to the input
                                if self.input_objects[each]:
                                    self.pattern[each] += (1 - self.pattern[each]) * self.pattern_lr
                                    self.pattern[each] = min(1, self.pattern[each])
                                else:
                                    self.pattern[each] -= self.pattern[each] * self.pattern_lr
                                    self.pattern[each] = max(0, self.pattern[each])
                            else:
                                # if I want to make my result "False", I need to make my pattern far from the input
                                if not self.input_objects[each]:
                                    self.pattern[each] += (1 - self.pattern[each]) * self.pattern_lr
                                    self.pattern[each] = min(1, self.pattern[each])
                                else:
                                    self.pattern[each] -= self.pattern[each] * self.pattern_lr
                                    self.pattern[each] = max(0, self.pattern[each])
                else:
                    # my pattern is bad
                    self.input_weights[each] *= self.weight_decay
                    if expected_value:
                        # if I want to make my result "True", I need to make my pattern close to the input
                        if self.input_objects[each]:
                            self.pattern[each] += (1 - self.pattern[each]) * self.pattern_lr
                            self.pattern[each] = min(1, self.pattern[each])
                        else:
                            self.pattern[each] -= self.pattern[each] * self.pattern_lr
                            self.pattern[each] = max(0, self.pattern[each])
                    else:
                        # if I want to make my result "False", I need to make my pattern far from the input
                        if not self.input_objects[each]:
                            self.pattern[each] += (1 - self.pattern[each]) * self.pattern_lr
                            self.pattern[each] = min(1, self.pattern[each])
                        else:
                            self.pattern[each] -= self.pattern[each] * self.pattern_lr
                            self.pattern[each] = max(0, self.pattern[each])
            else:  # my attention is bad
                self.input_weights[each] *= self.weight_decay
        self.weight_regularize()

    def weight_regularize(self):
        self.input_weights = list(np.array(self.input_weights) / max(self.input_weights))

    def boost(self):
        for each in self.indices:
            self.input_weights[each] *= self.award
        self.weight_regularize()
