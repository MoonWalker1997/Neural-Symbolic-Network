import tqdm

import pickle

import numpy as np

from NSN_2_1 import NSN_2
from Nodes.ONodes import ONodes

ONodes_1 = ONodes()
NSN_2 = NSN_2(ONodes_1)
num_bit = 10  # 1024 possible training cases
training = 700
testing = 300
epoch = 100


# structure, 10 inputs, 2 outputs


def logic_expression(num_bit):
    x = [np.random.choice([True, False], p=[0.5, 0.5]) for _ in range(num_bit)]
    y = [x[0] and x[6], not (x[0] and x[6])]
    return x, y


if __name__ == '__main__':
    NSN_2.show_structure()
    for _ in range(epoch):
        # training process
        for i in range(training):
            x, y = logic_expression(num_bit)
            NSN_2.forward(x)
            NSN_2.backward(y)
        NSN_2.analyze(True)
        # testing process
        succeed = 0
        fail = 0
        for i in range(testing):
            x, y = logic_expression(num_bit)
            NSN_2.forward(x)
            tmp = [each.value for each in NSN_2.output_layer.output_SubLayer.objects]
            if all(np.array(y) == np.array(tmp)):
                succeed += 1
            else:
                fail += 1

        print(succeed / (succeed + fail))
    NSN_2.save(
        "C:/Users/TORY/OneDrive - Temple University/AGI research/Neural Symbolic Network/models/NSN_2_model_1.json")
