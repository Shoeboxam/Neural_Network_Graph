from .Gate import *


class Logistic(Gate):
    @_cache
    def __call__(self, stimulus):
        return 1.0 / (1.0 + np.exp(-super().__call__(stimulus)))

    @_cache
    def gradient(self, stimulus, variable, grad):
        print(list(stimulus.values())[0].shape)
        return super().gradient(stimulus, variable, grad * self(stimulus) * (1.0 - self(stimulus)))
