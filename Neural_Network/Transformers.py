from .Gate import *
from .Variable import Variable


def _cache(method):
    def decorator(self, *args, **kwargs):
        if self._cached_stimulus == args[0]:
            self._cached_stimulus = args[0]
            return lambda **kw: getattr(self, '_cached_' + method.__name__)
        feature = method(self, *args, **kwargs)

        setattr(self, '_cached_' + method.__name__, feature)
        return feature

    return decorator


# Always return initial value computed by function
def _store(method):
    def decorator(self, **kwargs):
        if not getattr(self, '_stored_' + method.__name__):
            setattr(self, '_stored_' + method.__name__, method(self, **kwargs))
        return getattr(self, '_stored_' + method.__name__)
    return decorator


class Transform(Gate):
    def __init__(self, children, nodes):
        super().__init__(children)
        self._variables = {
            'weights': Variable(np.random.normal(size=(nodes, self.input_nodes))),
            'biases': Variable(np.zeros((nodes, 1)))}

    @property
    def output_nodes(self):
        return self._variables['weights'].shape[0]

    @_cache
    def __call__(self, stimulus):
        return self._variables['weights'] @ super().__call__(stimulus) + self._variables['biases']

    @_cache
    def gradient(self, stimulus, variable, grad):
        propagated = super().__call__(stimulus)
        if variable is self._variables['weights']:
            # Full derivative: This is a tensor product simplification to avoid the use of the kron product
            return grad.T @ propagated[None]
        if variable is self._variables['biases']:
            return grad
        return super().gradient(stimulus, variable, grad @ self._variables['weights'])


class TransformRecurrent(Transform):
    def __init__(self, children, nodes, decay=1.0):
        super().__init__(children, nodes)
        self.decay = decay
        self._variables['internal'] = Variable(np.random.normal(size=(nodes, nodes)))

        self._weight_gradient = np.zeros((self.output_nodes, self.input_nodes))
        self._bias_gradient = np.zeros((self.output_nodes, 1))
        self._internal_gradient = np.zeros((self.output_nodes, 1))

        self._prediction = np.zeros((self.output_nodes, 1))

    @_cache
    def __call__(self, stimulus):
        recurrent = self._variables['internal'] @ self._prediction
        propagated = np.vstack((super(super(), self).__call__(stimulus), self.decay * recurrent))

        self._prediction = self._variables['weights'] @ propagated + self._variables['biases']
        return self._prediction

    @_cache
    def gradient(self, stimulus, variable, grad):
        propagated = super(super(), self).__call__(stimulus)
        if variable is self._variables['weights']:
            return grad.T @ propagated[None] @ (1 + self._variables['internal'])
        if variable is self._variables['biases']:
            return grad @ (1 + self._variables['internal'])
        if variable is self._variables['internal']:
            return grad.T @ propagated[None]
        return super().gradient(stimulus, variable, grad @ self._variables['weights'])

    @property
    def input_nodes(self):
        return super().input_nodes + self.output_nodes
