import numpy as np

import NSN
from Nodes.CNode import CNode
from Nodes.IDENTITY_ONode import IDENTITY_ONode


class CompoundConvLayer:
    """
        A layer mainly does one thing - it will make all objects forward and backward.
        But these objects are not only stored in an array, they might have a data-structure, e.g., matrix.
        So, a layer is to define the data-structure of inside objects.

        A compound conv layer is used for compound convolution, but it is not the mathematical convolution and it is
        just a compounding processing, though a convolution-like sliding-window is used.

        Currently, only 3x3 stride-1 convolution is supported, and it is by default.

        Different from these symbolic-level layers, it is in feature-level, so its input is bounded by some categories,
        here, specifically the spatial category. And so, its objects have a spatial-bounded data-structure. This is
        achieved by self.set_objects()
    """

    def __init__(self, num_conv, shape):
        self.num_conv = num_conv
        self.objects = []
        self.idt_objects = []
        self.set_objects(shape)

    def set_objects(self, shape):
        """
            Say the input size is (c, i, j), and it is convoluted by n kernels, the size will be (n, i-2, j-2).

            Convoluted results at the same position have c references, each kernel will so have to select one from
            these c references with an identity node, and there are n kernel, the size will be (n, ., .).
        """

        # specifying these compounds nodes
        for c in range(shape[0]):
            tmp1 = []
            for i in range(shape[1] - 2):
                tmp2 = []
                for j in range(shape[2] - 2):
                    tmp_cpd = CNode()
                    tmp2.append(tmp_cpd)
                tmp1.append(tmp2)
            self.objects.append(tmp1)
        self.objects = np.array(self.objects)

        # specifying these identity nodes
        for n in range(self.num_conv):
            tmp1 = []
            for i in range(shape[1] - 2):
                tmp2 = []
                for j in range(shape[2] - 2):
                    tmp_cpd = IDENTITY_ONode()
                    tmp2.append(tmp_cpd)
                tmp1.append(tmp2)
            self.idt_objects.append(tmp1)
        self.idt_objects = np.array(self.idt_objects)

        # connect these compound nodes and the identity nodes
        for n in range(self.num_conv):
            for i in range(shape[1] - 2):
                for j in range(shape[2] - 2):
                    self.idt_objects[n, i, j].input_objects = list(self.objects[:, i, j].flatten())
                    self.idt_objects[n, i, j].input_weights = NSN.weights_initialization(shape[0])

    def forward(self):
        for c in range(self.objects.shape[0]):  # channels
            for i in range(self.objects.shape[1]):  # columns
                for j in range(self.objects.shape[2]):  # rows
                    self.objects[c, i, j].value = self.objects[c, i, j].forward()
        for n in range(self.idt_objects.shape[0]):  # num_kernels
            for i in range(self.idt_objects.shape[1]):  # columns
                for j in range(self.idt_objects.shape[2]):  # rows
                    self.idt_objects[n, i, j].value = self.idt_objects[n, i, j].forward()
