from ..Node import *

# using hadamard identity on elementwise functions for efficiency
# A @ diag(b) == A * b.T


class Sinusoidal(Node):
    def propagate(self, features):
        return np.sin(np.vstack(features))

    def backpropagate(self, features, variable, gradient):
        return gradient * np.swapaxes(np.cos(np.vstack(features)), -1, -2)


class Logistic(Node):
    def propagate(self, features):
        return 1.0 / (1.0 + np.exp(-np.vstack(features)))

    def backpropagate(self, features, variable, grad):
        forward = self.propagate(features)
        # d propagate(r) / d r = diag(propagate(r) * (1 - propagate(r)))
        return grad * np.swapaxes(forward * (1.0 - forward), -1, -2)


class Bent(Node):
    def propagate(self, features):
        return (np.sqrt(np.vstack(features)**2 + 1) - 1) / 2 + np.vstack(features)

    def backpropagate(self, features, variable, gradient):
        diag = (np.vstack(features) / (2*np.sqrt(np.vstack(features)**2 + 1)) + 1)
        return gradient * np.swapaxes(diag, -1, -2)


class Softmax(Node):
    def propagate(self, features):
        features = np.vstack(features)
        unnormalized = np.exp(features - features.max(axis=0))  # Max is used for numerical stability
        return unnormalized / np.sum(unnormalized)

    def backpropagate(self, features, variable, grad):
        features = np.vstack(features)
        return grad @ (diag_3d(self.propagate(features)) - features @ np.swapaxes(features, -1, -2))


def diag_3d(arr):
    # construct an n+1 dimensional empty matrix
    zeros = np.zeros((*arr.shape, arr.shape[-1]), arr.dtype)
    # assign arr to a writable view of the interior diagonal along the final axes
    np.einsum('...ii->...i', zeros)[...] = arr
    return zeros
