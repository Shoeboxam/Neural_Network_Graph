import abc
import numpy as np
from collections import defaultdict

# any class not listed in __all__ is not exported/available for use
__all__ = [
    # 'Optimizer',
    'GradientDescent',
    'Momentum',
    'Nesterov',
    'Adagrad',
    'RMSprop',
    'Adam',
    'Adamax',
    'Nadam',
    'Quickprop'
]


class Optimizer(object):
    def __init__(self, function, rate=None):
        self.function = function
        self.iteration = 0
        self.rate = Optimizer._default_hyperparameter(rate, .01)

    def iterate(self, stimulus):
        for var in self.function.variables:
            gradient = np.average(self.function.gradient(stimulus, var), axis=0)

            var += self.iterate_variable(var, stimulus, gradient)
            self.iteration += 1

    @abc.abstractmethod
    def iterate_variable(self, var, stimulus, gradient):
        pass

    @staticmethod
    def _default_hyperparameter(value, default):
        default_scalar = value if type(value) is dict else default
        hyperparameter = defaultdict(lambda: default_scalar)
        if type(value) is dict:
            for variable in value:
                hyperparameter[variable] = value[variable]
        return hyperparameter


class GradientDescent(Optimizer):
    def iterate_variable(self, var, stimulus, gradient):
        return -self.rate[var] * gradient


class Momentum(Optimizer):
    def __init__(self, function, decay=None, **kwargs):
        super().__init__(function, **kwargs)

        self.decay = Optimizer._default_hyperparameter(decay, .2)
        self.update = {var: np.zeros(var.shape) for var in self.function.variables}

    def iterate_variable(self, var, stimulus, gradient):
        self.update[var] = gradient + self.decay[var] * self.update[var]
        return -self.rate[var] * self.update[var]


class Nesterov(Optimizer):
    def __init__(self, function, decay=None, **kwargs):
        super().__init__(function, **kwargs)

        self.decay = Optimizer._default_hyperparameter(decay, .9)
        self.update = {var: np.zeros(var.shape) for var in self.function.variables}

    def iterate_variable(self, var, stimulus, gradient):
        # SECTION 3.5: https://arxiv.org/pdf/1212.0901v2.pdf
        update_old = self.update[var]

        self.update[var] = self.decay[var] * self.update[var] - self.rate[var] * gradient
        return +self.decay[var] * (self.update[var] - update_old) + self.update[var]


class Adagrad(Optimizer):
    def __init__(self, function, wedge=.001, decay=.9, **kwargs):
        super().__init__(function, **kwargs)

        self.decay = Optimizer._default_hyperparameter(decay, .9)
        self.wedge = Optimizer._default_hyperparameter(wedge, .001)

        self.grad_square = {var: np.zeros([var.shape[0]] * 2) for var in function.variables}

    def iterate_variable(self, var, stimulus, gradient):
        # Historical gradient with exponential decay
        self.grad_square[var] = self.decay[var] * self.grad_square[var] \
                                + (1 - self.decay[var]) * gradient @ gradient.T
        # Normalize gradient
        norm = (np.sqrt(np.diag(self.grad_square[var])) + self.wedge[var])[..., None]
        return -self.rate[var] * gradient / norm


class Adadelta(Optimizer):
    def __init__(self, function, decay=None, wedge=None, **kwargs):
        super().__init__(function, **kwargs)

        self.decay = Optimizer._default_hyperparameter(decay, .9)
        self.wedge = Optimizer._default_hyperparameter(wedge, 1e-8)

        self.update = {var: np.ones(var.shape) for var in function.variables}
        self.grad_square = {var: np.zeros([var.shape[0]] * 2) for var in function.variables}
        self.update_square = {var: np.zeros([var.shape[0]] * 2) for var in function.variables}

    def iterate_variable(self, var, stimulus, gradient):
        # TODO: This method is not converging
        # EQN 14: https://arxiv.org/pdf/1212.5701.pdf
        # Rate is derived
        self.grad_square[var] = self.decay[var] * self.grad_square[var] + \
                                (1 - self.decay[var]) * gradient @ gradient.T

        rate = np.sqrt(np.diag(self.update_square[var]) + self.wedge)[..., None]
        self.update[var] = -(rate / np.sqrt(np.diag(self.grad_square[var]) + self.wedge)[..., None]) * gradient
        # print((rate / np.sqrt(np.diag(self.grad_square[var]) + self.wedge)[..., None]).shape)

        # Prepare for next iteration
        self.update_square[var] = self.decay[var] * self.update_square[var] \
                                  + (1 - self.decay[var]) * self.update[var] @ self.update[var].T

        return -self.update[var]


class RMSprop(Optimizer):
    def __init__(self, function, decay=None, wedge=None, **kwargs):
        super().__init__(function, **kwargs)

        self.decay = Optimizer._default_hyperparameter(decay, .9)
        self.wedge = Optimizer._default_hyperparameter(wedge, 1e-8)

        self.grad_square = {var: np.zeros([var.shape[0]] * 2) for var in function.variables}

    def iterate_variable(self, var, stimulus, gradient):
        # SLIDE 29: http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
        self.grad_square[var] = self.decay[var] * self.grad_square[var] \
                                + (1 - self.decay[var]) * gradient @ gradient.T

        return -self.rate[var] * gradient / (np.sqrt(np.diag(self.grad_square[var])) + self.wedge[var])[..., None]


class Adam(Optimizer):
    def __init__(self, function, decay_first_moment=None, decay_second_moment=None, wedge=None, **kwargs):
        super().__init__(function, **kwargs)
        self.decay_first_moment = Optimizer._default_hyperparameter(decay_first_moment, .9)
        self.decay_second_moment = Optimizer._default_hyperparameter(decay_second_moment, .999)
        self.wedge = Optimizer._default_hyperparameter(wedge, 1e-8)

        self.grad_cache = {var: np.ones(var.shape) for var in function.variables}
        self.grad_square = {var: np.zeros([var.shape[0]] * 2) for var in function.variables}

    def iterate_variable(self, var, stimulus, gradient):
        # Adaptive moment estimation: https://arxiv.org/pdf/1412.6980.pdf4
        self.grad_cache[var] = self.decay_first_moment[var] * self.grad_cache[var] \
                               + (1 - self.decay_first_moment[var]) * gradient

        self.grad_square[var] = self.decay_second_moment[var] * self.grad_square[var] \
                                + (1 - self.decay_second_moment[var]) * gradient @ gradient.T

        first_moment = self.grad_cache[var] / (1 - self.decay_first_moment[var] ** self.iteration)
        second_moment = self.grad_square[var] / (1 - self.decay_second_moment[var] ** self.iteration)

        return -self.rate[var] * first_moment / (np.sqrt(np.diag(second_moment) + self.wedge[var])[..., None])


class Adamax(Optimizer):
    def __init__(self, function, decay_first_moment=None, decay_second_moment=None, **kwargs):
        super().__init__(function, **kwargs)

        self.decay_first_moment = Optimizer._default_hyperparameter(decay_first_moment, .9)
        self.decay_second_moment = Optimizer._default_hyperparameter(decay_second_moment, .999)

        self.grad_cache = {var: np.ones(var.shape) for var in function.variables}
        self.second_moment = {var: 0.0 for var in function.variables}

    def iterate_variable(self, var, stimulus, gradient):
        # Adaptive moment estimation: https://arxiv.org/pdf/1412.6980.pdf
        self.grad_cache[var] = self.decay_first_moment[var] * self.grad_cache[var] \
                               + (1 - self.decay_first_moment[var]) * gradient
        self.second_moment[var] = max(self.decay_second_moment[var] * self.second_moment[var],
                                      np.linalg.norm(gradient, ord=np.inf))

        first_moment = self.grad_cache[var] / (1 - self.decay_first_moment[var] ** self.iteration)
        return -self.rate[var] * first_moment / self.second_moment[var]


class Nadam(Optimizer):
    def __init__(self, function, decay_first_moment=None, decay_second_moment=None, wedge=None, **kwargs):
        super().__init__(function, **kwargs)
        self.decay_first_moment = Optimizer._default_hyperparameter(decay_first_moment, .9)
        self.decay_second_moment = Optimizer._default_hyperparameter(decay_second_moment, .999)
        self.wedge = Optimizer._default_hyperparameter(wedge, 1e-8)

        self.grad_cache = {var: np.ones(var.shape) for var in function.variables}
        self.grad_square = {var: np.zeros([var.shape[0]] * 2) for var in function.variables}

    def iterate_variable(self, var, stimulus, gradient):
        # Nesterov adaptive moment estimation: http://cs229.stanford.edu/proj2015/054_report.pdf
        self.grad_cache[var] = self.decay_first_moment[var] * self.grad_cache[var] \
                               + (1 - self.decay_first_moment[var]) * gradient
        self.grad_square[var] = self.decay_second_moment[var] * self.grad_square[var] \
                                + (1 - self.decay_second_moment[var]) * gradient @ gradient.T

        first_moment = self.grad_cache[var] / (1 - self.decay_first_moment[var] ** self.iteration)
        second_moment = self.grad_square[var] / (1 - self.decay_second_moment[var] ** self.iteration)

        nesterov = (self.decay_first_moment[var] * first_moment +
                    (1 - self.decay_first_moment[var]) * gradient /
                    (1 - self.decay_first_moment[var] ** self.iteration))
        return -self.rate[var] * nesterov / (np.sqrt(np.diag(second_moment) + self.wedge[var])[..., None])


class Quickprop(Optimizer):
    def __init__(self, function, maximum_growth_factor=None, **kwargs):
        super().__init__(function, **kwargs)
        self.maximum_growth_factor = Optimizer._default_hyperparameter(maximum_growth_factor, 1.2)

        self.update = {var: np.ones(var.shape) for var in function.variables}
        self.gradient_cache = {var: np.zeros(var.shape) for var in function.variables}

    def iterate_variable(self, var, stimulus, gradient):
        # https://arxiv.org/pdf/1606.04333.pdf
        limit = np.abs(self.update[var]) * self.maximum_growth_factor[var]
        self.update[var] = np.clip(gradient / (self.gradient_cache[var] - gradient), -limit, limit)

        self.gradient_cache[var] = gradient.copy()

        return -self.rate[var] * self.update[var]


class L_BFGS(Optimizer):
    def __init__(self, function, decay=None, **kwargs):
        super().__init__(function, **kwargs)
        self.decay = Optimizer._default_hyperparameter(decay, .2)

        self.update = {var: np.zeros(var.shape) for var in function.variables}
        self.grad_cache = {var: np.zeros(var.shape) for var in function.variables}

        self.hessian_inv = {var: np.eye(var.shape[0]) for var in function.variables}

    def iterate_variable(self, var, stimulus, gradient):
        """THIS METHOD IS NOT FULLY IMPLEMENTED"""

        # update_delta = self.update[var] - update
        # grad_delta = gradient - self.grad_cache[var]
        #
        # alpha = (update_delta.T @ gradient) / (grad_delta.T @ update_delta)
        #
        # self.update[var] = gradient + self.decay[var] * self.update[var]
        # return -self.rate[var] * self.update[var]
        pass
