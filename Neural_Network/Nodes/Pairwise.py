from Neural_Network.Node import Node
import numpy as np


class GaussianNeighborhood(Node):
    def propagate(self, features):
        neighborhood = np.exp(-np.square(features))
        return neighborhood / np.sum(neighborhood)

    def backpropagate(self, features, variable, gradient):
        neighborhood = -2 * np.exp(-np.square(features)) @ features
        return neighborhood / np.sum(neighborhood)


class PairwiseEuclidean(Node):
    def propagate(self, sample_space):
        # Singleton axes are added to end to force broadcasting. [f, 1, b] - [f, b, 1] -> [f, b, b]
        delta = np.sqrt(np.sum(np.square(sample_space[..., None] - sample_space[..., None, :]), axis=0))
        return delta / (2 * np.var(delta, axis=0))

    def backpropagate(self, sample_space, variable, gradient):
        return -(sample_space[..., None] - sample_space[..., None, :])


class PairwiseDot(Node):
    def propagate(self, sample_space):
        return np.einsum('i...,i...->...', sample_space[..., None] - sample_space[..., None, :])

    def backpropagate(self, feature, variable, gradient):
        pass
