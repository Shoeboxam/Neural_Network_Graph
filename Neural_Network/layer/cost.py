import abc

from config import NP_BACKEND
from Neural_Network.node import Node
if NP_BACKEND == 'NUMPY':
    import numpy as np
elif NP_BACKEND == 'JAX':
    import jax.numpy as np


# NOTE: The signatures of cost functions do not match the parent class
class Cost(Node):
    def __init__(self, predictions, expectations):
        # Set self.children
        super().__init__(predictions)

        # Set self.expectations
        if type(expectations) is not list:
            expectations = [expectations]
        self.expectations = expectations

    def __call__(self, stimulus):
        predictions = np.vstack([child(stimulus) for child in self.children])
        expectation = np.vstack([child(stimulus) for child in self.expectations])
        return self.propagate(predictions, expectation)

    def gradient(self, stimulus, variable):
        predictions = np.vstack([child(stimulus) for child in self.children])
        expectation = np.vstack([child(stimulus) for child in self.expectations])
        grad = self.backpropagate(predictions, expectation)

        return np.vstack([child.gradient(stimulus, variable, grad) for child in self.children])

    @abc.abstractmethod
    def propagate(self, prediction, expectation):
        pass

    @abc.abstractmethod
    def backpropagate(self, prediction, expectation):
        pass


class SumSquared(Cost):
    def propagate(self, prediction, expectation):
        return np.average((expectation - prediction)**2, axis=0)

    def backpropagate(self, prediction, expectation):
        return -2 * np.swapaxes(expectation - prediction, -1, -2)


class CrossEntropy(Cost):
    def propagate(self, prediction, expectation):
        return -np.average(expectation * np.log(prediction) + (1 - expectation) * np.log(1 - prediction))

    def backpropagate(self, prediction, expectation):
        return -(expectation - prediction) / (prediction * (1 - prediction))


class CrossEntropySoftmax(Cost):
    # Combination of cross entropy and softmax.
    # Train a classifier with CrossEntropySoftmax, and then add a Softmax to the trained network
    def propagate(self, prediction, expectation):
        return -np.average(expectation * np.log(prediction))

    def backpropagate(self, prediction, expectation):
        return -(expectation - prediction)


class AngularSeparation(Cost):
    def propagate(self, prediction, expectation):
        return np.arccos(expectation.T @ prediction)

    def backpropagate(self, prediction, expectation):
        return -expectation / np.sqrt(1 + expectation.T @ prediction)


class KullbackLiebler(Cost):
    def propagate(self, prediction, expectation):
        return np.sum(prediction @ np.log(prediction / expectation))

    def backpropagate(self, prediction, expectation):
        return -prediction / expectation
