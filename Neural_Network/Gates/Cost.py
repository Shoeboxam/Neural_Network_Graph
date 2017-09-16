from Neural_Network.Gate import *
import numpy as np

# NOTE: The signatures of cost functions do not match the parent class


class SumSquared(Gate):
    def __init__(self, children, environment):
        super().__init__(children)
        self.environment = environment

    @cache
    def propagate(self, predicted, expected):
        return np.average((expected[self.environment.tag] - features)**2, axis=0)

    @cache
    def backpropagate(self, predicted, variable, expected):
        return -2 * (expected[self.environment.tag] - predicted)


class CrossEntropy(Gate):
    def __init__(self, children, environment):
        super().__init__(children)
        self.environment = environment

    @cache
    def propagate(self, predicted, expected):
        expect = expected[self.environment.tag]
        return -np.average(expect * np.log(predicted) + (1 - expect) * np.log(1 - predicted))

    @cache
    def backpropagate(self, predicted, variable, expected):
        expect = expected[self.environment.tag]
        return -(expect - predicted) / (predicted * (1 - predicted))


class CrossEntropySoftmax(Gate):
    # Combination of cross entropy and softmax.
    # Train a classifier with CrossEntropySoftmax, and then add a Softmax to the trained network
    def __init__(self, children, environment):
        super().__init__(children)
        self.environment = environment

    @cache
    def propagate(self, predicted, expected):
        return -np.average(expected[self.environment.tag] * np.log(predicted))

    @cache
    def gradient(self, predicted, variable, expected):
        return -(expected[self.environment.tag] - predicted)


class AngularSeparation(Gate):
    def __init__(self, children, environment):
        super().__init__(children)
        self.environment = environment

    @cache
    def propagate(self, predicted, expected):
        return np.arccos(expected[self.environment.tag].T @ predicted)

    @cache
    def gradient(self, predicted, variable, expected):
        return -expected[self.environment.tag] / np.sqrt(1 + expected[self.environment.tag].T @ predicted)


class KullbackLiebler(Gate):
    def __init__(self, children, environment):
        super().__init__(children)
        self.environment = environment

    @cache
    def propagate(self, predicted, expected):
        return np.sum(predicted @ np.log(predicted / expected))

    @cache
    def gradient(self, predicted, variable, expected):
        return -predicted / expected
