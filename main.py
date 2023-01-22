import time

import numpy as np

from Nodes.ONodes import ONodes
from models.Propositional_logic_NSN_10_5_3_1 import Propositional_Logic_NSN
from util.LogicExpressionTrainer import LogicExpressionTrainer

ONodes = ONodes()
NSN = Propositional_Logic_NSN(ONodes)

# structure, 3 inputs, 3 hidden nodes, 1 output

class LET(LogicExpressionTrainer):

    def __init__(self, num_bit, num_training, num_testing):
        super(LET, self).__init__(num_bit, num_training, num_testing)

    def expression(self, x):
        return [
            (not x[0] or x[2]) and x[4] and x[3] and not x[1]
        ]

num_bit = 10
num_training = 4
num_testing = 2000

num_epoch = 5

LET = LET(num_bit, num_training, num_testing)

if __name__ == '__main__':
    NSN.draw()
    for _ in range(num_epoch):
        # training process
        for i in range(num_training):
            x, y = LET.produce(training=True)
            NSN.forward(x)
            NSN.backward(y)
        NSN.draw()
        # time.sleep(3)
        # testing process
        succeed = 0
        fail = 0
        for i in range(num_testing):
            x, y = LET.produce(training=False)
            NSN.forward(x)
            if np.array(y) == NSN.output_layer.objects[0].value:
                succeed += 1
            else:
                fail += 1
        # resetting
        LET.repeat(shuffle=False)

        print(succeed / (succeed + fail))
