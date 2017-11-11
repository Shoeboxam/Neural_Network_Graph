from ..Gate import *


class Logistic(Gate):
    def propagate(self, features):
        return 1.0 / (1.0 + np.exp(-np.vstack(features)))

    def backpropagate(self, features, variable, grad):
        return grad * self.propagate(features) * (1.0 - self.propagate(features))


class Softmax(Gate):
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
