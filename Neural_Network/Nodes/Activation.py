from ..Node import *

# using hadamard identity on elementwise functions for efficiency
# A @ diag(b) == A * b.T


class Identity(Node):
    def propagate(self, features):
        return np.vstack(features)

    def backpropagate(self, features, variable, gradient):
        return gradient


class Heaviside(Node):
    def propagate(self, features):
        return piecewise(np.vstack(features), 0, 1)

    def backpropagate(self, features, variable, gradient):
        return gradient * 0


class ReLu(Node):
    def __init__(self, children, alpha=0):
        self.alpha = alpha
        super().__init__(children=children)

    def propagate(self, features):
        features = np.vstack(features)
        return piecewise(features, self.alpha * features, features)

    def backpropagate(self, features, variable, gradient):
        return gradient * np.swapaxes(piecewise(np.vstack(features), self.alpha, 1), -1, -2)


class Exponent(Node):
    def __init__(self, children, alpha=0):
        self.alpha = alpha
        super().__init__(children=children)

    def propagate(self, features):
        features = np.vstack(features)
        return piecewise(features, self.alpha*(np.exp(features) - 1), features)

    def backpropagate(self, features, variable, gradient):
        features = np.vstack(features)
        return gradient * np.swapaxes(piecewise(features, self.alpha * np.exp(features), 1), -1, -2)


class Softplus(Node):
    def propagate(self, features):
        return np.log(1 + np.exp(np.vstack(features)))

    def backpropagate(self, features, variable, gradient):
        return gradient * np.swapaxes((1 + np.exp(-np.vstack(features)))**-1, -1, -2)


class Gaussian(Node):
    def propagate(self, features):
        return np.exp(-np.vstack(features)**2)

    def backpropagate(self, features, variable, gradient):
        features = np.vstack(features)
        return gradient * np.swapaxes(-2 * features * np.exp(-features**2), -1, -2)


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


class Tanh(Node):
    def propagate(self, features):
        return np.tanh(np.vstack(features))

    def backpropagate(self, features, variable, gradient):
        return gradient * np.swapaxes(1 - self.propagate(features)**2, -1, -2)


class ArcTan(Node):
    def propagate(self, features):
        return np.arctan(np.vstack(features))

    def backpropagate(self, features, variable, gradient):
        return gradient * np.swapaxes(1 / (np.vstack(features)**2 + 1))


class SoftSign(Node):
    def propagate(self, features):
        features = np.vstack(features)
        return features / (1 + np.abs(features))

    def backpropagate(self, features, variable, gradient):
        return gradient * np.swapaxes(1 / (1 + np.abs(np.vstack(features)))**2, -1, -2)


class Log(Node):
    def propagate(self, features):
        features = np.vstack(features)
        return piecewise(features, np.log(1 + features), -np.log(1 - features))

    def backpropagate(self, features, variable, gradient):
        features = np.vstack(features)
        return gradient * np.swapaxes(piecewise(features, 1 / (1 + features), 1 / (1 - features), -1, -2))


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


def piecewise(x, lower, upper, thresh=0):

    low_indices = np.where(x < thresh)
    if type(lower) == float or type(lower) == int:
        x[low_indices] = lower
    else:
        x[low_indices] = lower[low_indices]

    up_indices = np.where(x > thresh)
    if type(upper) == float or type(upper) == int:
        x[up_indices] = upper
    else:
        x[up_indices] = upper[up_indices]
    return x