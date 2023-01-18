import numpy as np

"""
A class for nodes except for the input nodes.
"""


class Node:

    def __init__(self):
        self.input_weights = []  # a list of input weights
        self.input_objects = []  # a list of input objects
        # self.input_weights[i] is the weight of self.input_objects[i]
