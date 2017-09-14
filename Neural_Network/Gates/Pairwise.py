from .Gate import Gate
import numpy as np


class PairwiseEuclidean(Gate):
    def propagate(self, sample_space):
        # Singleton axes are added to end to force broadcasting. [f, 1, b] - [f, b, 1] -> [f, b, b]
        delta = np.sqrt(np.sum(np.square(sample_space[..., :, None] - sample_space[..., None]), axis=0))
        return delta / (2 * np.var(delta, axis=0))

    def backpropagate(self, features, variable, gradient):
        pass


# This was suggested as a possible distance measure for binary stimuli, but I'm skeptical... what about two ones?
class PairwiseDot(Gate):
    def propagate(self, sample_space):
        return np.einsum('i...,i...->...', sample_space[..., :, None] - sample_space[..., None])

    def backpropagate(self, feature, variable, gradient):
        pass
