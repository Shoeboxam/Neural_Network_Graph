from .Gate import *
import numpy as np

# NOTE: The signatures of cost functions do not match the parent class


class SumSquared(Gate):
    def __init__(self, children, environment):
        super().__init__(children)
        self.environment = environment

    @cache
    def __call__(self, stimulus, expected):
        return np.average((expected[self.environment.tag] - super().__call__(stimulus))**2, axis=0)

    @cache
    def gradient(self, stimulus, variable, expected):
        return super().gradient(stimulus, variable, -2 * (expected[self.environment.tag] - super().__call__(stimulus)))


class CrossEntropy(Gate):
    def __init__(self, children, environment):
        super().__init__(children)
        self.environment = environment

    @cache
    def __call__(self, stimulus, expected):
        expect = expected[self.environment.tag]
        predict = super().__call__(stimulus)
        return -np.average(expect * np.log(predict) + (1 - expect) * np.log(1 - predict))

    @cache
    def gradient(self, stimulus, variable, expected):
        expect = expected[self.environment.tag]
        predict = super().__call__(stimulus)
        return super().gradient(stimulus, variable, -(expect - predict) / (predict * (1 - predict)))


class CrossEntropySoftmax(Gate):
    # Combination of cross entropy and softmax.
    # Train a classifier with CrossEntropySoftmax, and then add a Softmax to the trained network
    def __init__(self, children, environment):
        super().__init__(children)
        self.environment = environment

    @cache
    def __call__(self, stimulus, expected):
        return -np.average(expected[self.environment.tag] * np.log(super().__call__(stimulus)))

    @cache
    def __call__(self, stimulus, variable, expected):
        return super().gradient(stimulus, variable, -(expected[self.environment.tag] - super().__call__(stimulus)))