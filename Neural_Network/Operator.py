import numpy as np
from .Gate import Gate

_scalar = [str, int, float]


class Add(Gate):
    def propagate(self, features):
        left = features[0]
        right = features[1]

        if type(left) in _scalar or type(right) in _scalar:
            return left + right

        # Implicitly cast lesser operand to a higher conformable dimension
        # Stimuli become vectorized, but bias units remain 1D. To add wx + b, must broadcast
        elif left.ndim == 2 and right.ndim == 1:
            return np.add(left, np.tile(right[..., np.newaxis], left.shape[1]))
        elif left.ndim == 1 and right.ndim == 2:
            return np.add(np.tile(left[..., np.newaxis], right.shape[1]), right)

        elif left.ndim == 3 and right.ndim == 2:
            return np.add(left, np.tile(right[..., np.newaxis], left.shape[2]))
        elif left.ndim == 2 and right.ndim == 3:
            return np.add(np.tile(left[..., np.newaxis], right.shape[2]), right)
        else:
            return left + right

    def backpropagate(self, features, variable, gradient):
        derivatives = {}

        for child in self.children:
            if variable is child or variable in child.variables:
                derivatives[child] = gradient
        return derivatives


class Matmul(Gate):
    @property
    def output_nodes(self):
        return self.children[0].output_nodes

    def propagate(self, features):
        left = features[0]
        right = features[0]

        if type(left) in _scalar or type(right) in _scalar:
            return left * right

        # Implicitly broadcast and vectorize matrix multiplication along axis 3
        # Matrix multiplication between 3D arrays is the matrix multiplication between respective matrix slices
        if left.ndim > 2 or right.ndim > 2:
            return np.einsum('ij...,jl...->il...', left, right)
        else:
            return left @ right

    def backpropagate(self, features, variable, gradient):
        derivatives = {}

        if variable is self.children[0]:
            derivatives[self.children[0]] = gradient.T @ features[1][None]
        elif variable in self.children[0].variables:
            derivatives[self.children[0]] = gradient @ features[1]

        if variable is self.children[1]:
            derivatives[self.children[1]] = gradient.T @ features[0][None]
        elif variable in self.children[1].variables:
            derivatives[self.children[1]] = gradient @ features[0]

        return derivatives
