import numpy as np

from NSN_3 import NSN_3
from Nodes.ONodes import ONodes

ONodes_1 = ONodes()
NSN_3 = NSN_3(ONodes_1)
num_bit = 3


# structure, 3 inputs, 1 outputs


def logic_expression(num_bit):
    x = [np.random.choice([True, False], p=[0.5, 0.5]) for _ in range(num_bit)]
    y = [x[0] and x[1]]
    return x, y


if __name__ == '__main__':
    NSN_3.analyze(True)
    NSN_3.show_structure()
    for _ in range(20):
        # training process
        for i in range(4):
            x, y = logic_expression(num_bit)
            NSN_3.forward(x)
            # NSN_3.structure()
            NSN_3.backward(y)
            # NSN_3.structure()
        NSN_3.analyze(True)
        # testing process
        succeed = 0
        fail = 0
        for i in range(8):
            x, y = logic_expression(num_bit)
            NSN_3.forward(x)
            if np.array(y) == NSN_3.output_layer.output_SubLayer.objects[0].value:
                succeed += 1
            else:
                fail += 1

        print(succeed / (succeed + fail))
    # NSN_3.save(
    #     "C:/Users/TORY/OneDrive - Temple University/AGI research/Neural Symbolic Network/models/NSN_3_model_1.npy")
