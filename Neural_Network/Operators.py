import numpy as np
from .Gate import Gate

_scalar = [str, int, float]


class Add(Gate):
    def __call__(self, stimulus, parent=None):
        left = self.children[0](stimulus, self)
        right = self.children[1](stimulus, self)

        if type(left) in _scalar or type(right) in _scalar:
            features = left + right

        # Implicitly cast lesser operand to a higher conformable dimension
        # Stimuli become vectorized, but bias units remain 1D. To add wx + b, must broadcast
        elif left.ndim == 2 and right.ndim == 1:
            features = np.add(left, np.tile(right[..., np.newaxis], left.shape[1]))
        elif left.ndim == 1 and right.ndim == 2:
            features = np.add(np.tile(left[..., np.newaxis], right.shape[1]), right)

        elif left.ndim == 3 and right.ndim == 2:
            features = np.add(left, np.tile(right[..., np.newaxis], left.shape[2]))
        elif left.ndim == 2 and right.ndim == 3:
            features = np.add(np.tile(left[..., np.newaxis], right.shape[2]), right)
        else:
            features = left + right

        # Split a feature vector with respect to multiple parents
        if parent in self.parents:
            cursor = 0

            for par in self.parents:
                if parent is par:
                    break
                cursor += par.input_nodes
            features = features[cursor:cursor + parent.input_nodes]
        return features

    def gradient(self, stimulus, variable, grad):
        if variable in self._variables:
            return grad

        # The jacobian is the identity, so continue down the tree
        # Gradients passed along a branch need to be sliced to the portion relevant to the child
        gradients = []
        cursor = 0
        for child in self.children:
            gradients.append(child.gradient(stimulus, variable, grad[:, cursor:cursor + child.output_nodes]))
            cursor += child.output_nodes
        return np.vstack(gradients)


class Matmul(Gate):
    @property
    def output_nodes(self):
        return self.children[0].output_nodes

    # Forward pass
    def __call__(self, stimulus, parent=None):
        left = self.children[0](stimulus, self)
        right = self.children[1](stimulus, self)

        if type(left) in _scalar or type(right) in _scalar:
            return left * right

        # Implicitly broadcast and vectorize matrix multiplication along axis 3
        # Matrix multiplication between 3D arrays is the matrix multiplication between respective matrix slices
        if left.ndim > 2 or right.ndim > 2:
            features = np.einsum('ij...,jl...->il...', left, right)
        else:
            features = left @ right

        # Split a feature vector with respect to multiple parents
        if parent in self.parents:
            cursor = 0

            for par in self.parents:
                if parent is par:
                    break
                cursor += par.input_nodes
            features = features[cursor:cursor + parent.input_nodes]
        return features

    def gradient(self, stimulus, variable, grad):
        if variable is self.children[0]:
            return grad.T @ self.children[0](stimulus, self)[None]
        elif variable is self.children[1]:
            return grad.T @ self.children[1](stimulus, self)[None]
        else:
            if variable in self.children[0].variables:
                return self.children[1](self, stimulus)
            else:
                return self.children[0](self, stimulus)
