from ..Gate import *
import numpy as np


# NOTE: The signatures of cost functions do not match the parent class
class Cost(Gate):
    def __init__(self, predictions, expectations):
        super().__init__(predictions)
        if type(expectations) is not list:
            expectations = [expectations]
        self.expectations = expectations

    def __call__(self, stimulus, expected, variable):
        prediction = self.propagate([child(stimulus, self) for child in self.children])
        expectation = expected([child(expected, self) for child in self.expectations])
        return self.propagate(prediction, expectation)

    def gradient(self, stimulus, expected, variable):

        prediction = [child(stimulus, self) for child in self.children]
        expectation = expected([child(expected, self) for child in self.expectations])
        return self.backpropagate(prediction, expectation)

    def propagate(self, prediction, expectation):
        raise NotImplementedError("Cost is an abstract base class, and propagate is not defined.")

    def backpropagate(self, prediction, expectation):
        raise NotImplementedError("Cost is an abstract base class, and backpropagate is not defined.")


class SumSquared(Cost):
    @cache
    def propagate(self, prediction, expectation):
        return np.average((expectation - prediction)**2, axis=0)

    @cache
    def backpropagate(self, prediction, expectation):
        return -2 * (expectation - prediction)


class CrossEntropy(Cost):
    @cache
    def propagate(self, prediction, expectation):
        return -np.average(expectation * np.log(prediction) + (1 - expectation) * np.log(1 - prediction))

    @cache
    def backpropagate(self, prediction, expectation):
        return -(expectation - prediction) / (prediction * (1 - prediction))


class CrossEntropySoftmax(Cost):
    # Combination of cross entropy and softmax.
    # Train a classifier with CrossEntropySoftmax, and then add a Softmax to the trained network
    @cache
    def propagate(self, prediction, expectation):
        return -np.average(expectation * np.log(prediction))

    @cache
    def backpropagate(self, prediction, expectation):
        return -(expectation - prediction)


class AngularSeparation(Cost):
    @cache
    def propagate(self, prediction, expectation):
        return np.arccos(expectation.T @ prediction)

    @cache
    def backpropagate(self, prediction, expectation):
        return -expectation / np.sqrt(1 + expectation.T @ prediction)


class KullbackLiebler(Cost):
    @cache
    def propagate(self, prediction, expectation):
        return np.sum(prediction @ np.log(prediction / expectation))

    @cache
    def backpropagate(self, prediction, expectation):
        return -prediction / expectation
