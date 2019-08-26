import Neural_Network.optimizer as optimizers
import numpy as np


# differentially private optimization
# https://arxiv.org/pdf/1607.00133.pdf
class DPMixin(optimizers.Optimizer):
    def __init__(self, function, epsilon, delta, clipping_interval, num_rows, **kwargs):
        super().__init__(function, **kwargs)
        self.clipping_interval = clipping_interval
        self.epsilon = epsilon
        self.delta = delta

        self.num_rows = num_rows

        self.epsilon_used = 0

    def iterate(self, stimulus):
        batch_size = list(stimulus.values())[0].shape[0]
        sensitivity = 2 * self.clipping_interval / batch_size

        for var in self.function.variables:
            noise = DPMixin._gaussian_noise(var.shape, sensitivity, self.epsilon, self.delta)

            gradient = self.function.gradient(stimulus, var)
            np.clip(gradient, -self.clipping_interval, self.clipping_interval, out=gradient)
            gradient = np.average(gradient, axis=0) + noise

            var += self.iterate_variable(var, stimulus, gradient)

        self.iteration += 1
        self.epsilon_used += batch_size / self.num_rows

    @staticmethod
    def _gaussian_noise(size, sensitivity, epsilon, delta):
        sigma = np.sqrt(2*np.log(1.25/delta)) / epsilon
        return np.random.normal(size=size, scale=sensitivity * sigma)


def make_private_optimizer(cls_optimizer, **private_args):
    class DPOptimizer(DPMixin, cls_optimizer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **{**private_args, **kwargs})

    return DPOptimizer