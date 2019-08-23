from ..Node import *

# using hadamard identity on elementwise functions for efficiency
# A @ diag(b) == A * b.T


class Sinusoidal(Node):
    def propagate(self, features):
        return np.sin(np.vstack(features))

    def backpropagate(self, features, variable, gradient):
        print(gradient.shape)
        print(np.vstack(features).shape)
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
        unnormalized = np.exp(features - features.max())  # Max is used for numerical stability
        return unnormalized / np.sum(unnormalized)

    def backpropagate(self, features, variable, grad):
        return grad @ (diag_3d(self.propagate(features)) - features @ features.T)


def diag_3d(self):
    if self.ndim == 1:
        return np.diag(self)
    else:
        elements = []
        for idx in range(self.shape[-1]):
            elements.append(self[..., idx].diag())
        return np.stack(elements, self.ndim)
