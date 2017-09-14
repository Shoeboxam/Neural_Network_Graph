from .Gate import *
import numpy as np

# NOTE: The signatures of cost functions do not match the parent class


class SumSquared(Gate):
    def __init__(self, children, environment):
        super().__init__(children)
        self.environment = environment

    @cache
    def propagate(self, features, expected):
        return np.average((expected[self.environment.tag] - features)**2, axis=0)

    @cache
    def backpropagate(self, features, variable, expected):
        return -2 * (expected[self.environment.tag] - features)


class CrossEntropy(Gate):
    def __init__(self, children, environment):
        super().__init__(children)
        self.environment = environment

    @cache
    def propagate(self, features, expected):
        expect = expected[self.environment.tag]
        return -np.average(expect * np.log(features) + (1 - expect) * np.log(1 - features))

    @cache
    def backpropagate(self, features, variable, expected):
        expect = expected[self.environment.tag]
        return -(expect - features) / (features * (1 - features))


class CrossEntropySoftmax(Gate):
    # Combination of cross entropy and softmax.
    # Train a classifier with CrossEntropySoftmax, and then add a Softmax to the trained network
    def __init__(self, children, environment):
        super().__init__(children)
        self.environment = environment

    @cache
    def propagate(self, features, expected):
        return -np.average(expected[self.environment.tag] * np.log(features))

    @cache
    def gradient(self, features, variable, expected):
        return -(expected[self.environment.tag] - features)


class AngularSeparation(Gate):
    def __init__(self, children, environment):
        super().__init__(children)
        self.environment = environment

    @cache
    def propagate(self, features, expected):
        return np.arccos(expected[self.environment.tag].T @ features)

    @cache
    def gradient(self, features, variable, expected):
        return -expected[self.environment.tag] / np.sqrt(1 + expected[self.environment.tag].T @ features)
