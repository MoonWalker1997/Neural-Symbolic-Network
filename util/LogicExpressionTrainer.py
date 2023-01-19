import random
from abc import abstractmethod


class LogicExpressionTrainer:

    def __init__(self, num_bit, num_training, num_testing):
        self.num_bit = num_bit  # totally 2^num_bit expressions
        self.training = []
        self.num_training = num_training
        self.testing = []
        self.num_testing = num_testing
        self.generate()

    def generate(self):
        for i in range(self.num_training + self.num_testing):
            self.training.append(
                [True if each == 1 else False for each in ("{:0" + str(self.num_bit) + "b}").format(i)])
        self.testing = self.training[self.num_training:self.num_training + self.num_testing]
        self.training = self.training[:self.num_training]
        self.training_i = iter(self.training)
        self.testing_i = iter(self.testing)

    def produce(self, training = False):
        if training:
            tmp = next(self.training_i)
            return tmp, self.expression(tmp)
        else:
            tmp = next(self.testing_i)
            return tmp, self.expression(tmp)

    def repeat(self, shuffle = False):
        if shuffle:
            random.shuffle(self.training)
            random.shuffle(self.testing)
        self.training_i = iter(self.training)
        self.testing_i = iter(self.testing)

    @abstractmethod
    def expression(self, x):
        pass
