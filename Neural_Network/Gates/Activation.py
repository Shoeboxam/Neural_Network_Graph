from .Gate import *


class Logistic(Gate):
    @cache
    def __call__(self, stimulus):
        return 1.0 / (1.0 + np.exp(-super().__call__(stimulus)))

    @cache
    def gradient(self, stimulus, variable, grad):
        return super().gradient(stimulus, variable, grad * self(stimulus) * (1.0 - self(stimulus)))


class Softmax(Gate):
    @cache
    def __call__(self, stimulus):
        unnormalized = np.exp(super().__call__(stimulus) - super().__call__(stimulus).max())
        return unnormalized / np.sum(unnormalized)

    @cache
    def gradient(self, stimulus, variable, grad):
        jacobian = diag_3d(self(stimulus)) - super().__call__(stimulus) @ super().__call__(stimulus).T
        return super().gradient(stimulus, variable, grad @ jacobian)


def diag_3d(self):
    if self.ndim == 1:
        return np.diag(self)
    else:
        elements = []
        for idx in range(self.shape[-1]):
            elements.append(self[..., idx].diag())
        return np.stack(elements, self.ndim)
