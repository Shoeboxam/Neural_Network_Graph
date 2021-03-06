import abc

from config import NP_BACKEND

if NP_BACKEND == 'NUMPY':
    import numpy as np
elif NP_BACKEND == 'JAX':
    import jax.numpy as np


_scalar = [str, int, float]


# Return cached value if already computed for current stimulus
def cache(method):
    def decorator(self, stimulus, *args, **kwargs):
        if getattr(self, '_cached_' + method.__name__ + '_id') == id(stimulus):
            return getattr(self, '_cached_' + method.__name__)
        feature = method(self, stimulus, *args, **kwargs)

        setattr(self, '_cached_' + method.__name__ + '_id', id(stimulus))
        setattr(self, '_cached_' + method.__name__, feature)
        return feature

    return decorator


# Always return initial value computed by function
def store(method):
    def decorator(self, **kwargs):
        if not getattr(self, '_stored_' + method.__name__):
            setattr(self, '_stored_' + method.__name__, method(self, **kwargs))
        return getattr(self, '_stored_' + method.__name__)
    return decorator


class Node(object):
    def __init__(self, children):
        if type(children) not in [list, tuple]:
            children = [children]
        self.children = children

        # self._stored_variables = None
        # self._stored_output_nodes = None

        self._cached___call___id = 0
        self._cached___call__ = None

    @cache
    def __call__(self, stimulus):
        return self.propagate([child(stimulus) if issubclass(type(child), Node) else child for child in self.children])

    def gradient(self, stimulus, variable, grad):
        features = [child(stimulus) for child in self.children]

        # Holds derivatives for each branch of the function graph at the current node
        branches = []
        for child in self.children:
            # Derivative is complete
            if variable is child:
                branches.append(self.backpropagate(features, variable, grad))

            # Derivative needs to be further backpropagated
            elif variable in child.variables:
                derivative = self.backpropagate(features, variable, grad)
                branches.append(child.gradient(stimulus, variable, derivative))

        return sum(branches)

    # Define propagation in child classes
    @abc.abstractmethod
    def propagate(self, features):
        pass

    @abc.abstractmethod
    def backpropagate(self, features, variable, gradient):
        pass

    @property
    # @store
    def output_nodes(self):
        return sum([child.output_nodes for child in self.children])

    @property
    # @store
    def variables(self):
        """List the input variables"""
        return [variable for child in self.children if issubclass(type(child), Node)
                for variable in child.variables]

    @property
    def T(self):
        return Transpose(self)

    def __matmul__(self, other):
        return Matmul((self, other))

    def __add__(self, other):
        return Add((self, other))

    def __sub__(self, other):
        return Sub((self, other))

    def __pos__(self, other):
        return self

    def __neg__(self):
        return Neg(self)

    def __mul__(self, other):
        return Mul((self, other))

    def __truediv__(self, other):
        return Div((self, other))

    def __pow__(self, power, modulo=None):
        return Pow((self, power))

    def __iter__(self):
        return iter(self.variables)

    def __contains__(self, item):
        return item in self.variables

    def __str__(self):
        return self.__class__.__name__ + '(' + ','.join([str(child) for child in self.children]) + ')'


class Variable(Node):
    # prior versions inherited from np.ndarray. Switched to composition for jax compatibility
    def __init__(self, value, fixed=False, label=None):
        super().__init__(children=[])
        self.value = value
        self.fixed = fixed
        self.label = label or id(self)

    # Variable is a leaf in the network, so branching and pass-through logic is unnecessary and overriden
    def __call__(self, stimulus, parent=None):
        return self.propagate(stimulus)

    def propagate(self, stimulus):
        return self.value

    def backpropagate(self, features, variable, grad):
        if variable is self:
            return np.eye(self.output_nodes)
        return np.zeros([self.output_nodes] * 2)

    @property
    def output_nodes(self):
        return self.shape[0]

    @property
    def shape(self):
        return self.value.shape

    @property
    def variables(self):
        return [] if self.fixed else [self]

    # Variables are, by design, only equivalent if they share the same reference
    # If two different variables have all identical values, they are *still* not equal
    def __eq__(self, other):
        return self is other

    def __hash__(self):
        return id(self)

    def __str__(self):
        return f'[{"x".join([str(shape) for shape in self.value.shape])}]'

    def __iadd__(self, other):
        self.value += other

    def __isub__(self, other):
        self.value -= other

    def __imul__(self, other):
        self.value *= other

    def __imatmul__(self, other):
        self.value @= other

    def __idiv__(self, other):
        self.value /= other

    def __ifloordiv__(self, other):
        self.value //= other

    def __itruediv__(self, other):
        self.value /= other

    def __ipow__(self, other):
        self.value **= other

    def __imod__(self, other):
        self.value %= other


# ~~~~~~~~~~~~~~~~~~~~~
# Elementary operations
# ~~~~~~~~~~~~~~~~~~~~~

class Add(Node):
    def propagate(self, features):
        return features[0] + features[1]

    def backpropagate(self, features, variable, gradient):
        if variable in self.children:
            return np.swapaxes(gradient, -1, -2)
        return gradient

    def __str__(self):
        return ' + '.join([str(child) for child in self.children])

    @property
    def output_nodes(self):
        return self.children[0].output_nodes


class Sub(Node):
    def propagate(self, features):
        return features[0] - features[1]

    def backpropagate(self, features, variable, gradient):
        if variable in self.children[0]:
            return gradient
        else:
            return -gradient

    @property
    def output_nodes(self):
        return self.children[0].output_nodes

    def __str__(self):
        return ' - '.join([str(child) for child in self.children])


class Neg(Node):
    def propagate(self, features):
        return -np.vstack(features)

    def backpropagate(self, features, variable, gradient):
        return -gradient

    def __str__(self):
        return ' -' + str(self.children[0])


# Hadamard product
class Mul(Node):
    def propagate(self, features):
        return np.prod(features, axis=0)

    # TODO: variadic mul derivative
    def backpropagate(self, features, variable, gradient):
        # Take derivative of left side
        if variable in self.children[0]:
            return gradient * features[1].T

        # Take derivative of right side
        if variable in self.children[1]:
            return features[0].T * gradient

    def __str__(self):
        return ' * '.join([str(child) for child in self.children])


# Elementwise division, not inversion
class Div(Node):
    def propagate(self, features):
        return np.divide(features[0], features[1])

    def backpropagate(self, features, variable, gradient):
        # Take derivative of left side
        if variable in self.children[0]:
            return np.divide(gradient, features[1].T)

        # Take derivative of right side
        if variable in self.children[1]:
            return np.divide(features[0].T, gradient)

    def __str__(self):
        return str(self.children[0]) + ' / ' + str(self.children[1])


class Pow(Node):
    def propagate(self, features):
        return np.power(features[0], features[1])

    def backpropagate(self, features, variable, gradient):
        # Take derivative of base
        if variable in self.children[0]:
            return features[1] * np.power(features[0], features[1] - 1)

        # Take derivative of exponent
        if variable in self.children[1]:
            return np.log(features[1]) * np.power(features[0], features[1])

    def __str__(self):
        return str(self.children[0]) + '**' + str(self.children[1])


# Matrix product
class Matmul(Node):
    @property
    def output_nodes(self):
        return self.children[0].output_nodes

    # Matrix multiplication between N-dim arrays is the matrix multiplication across the tail axes
    # contraction occurs along:
    # left: last index
    # right: if 1D, then along only index. Else along second to last index

    def propagate(self, features):
        return features[0] @ features[1]

    def backpropagate(self, features, variable, gradient):

        # print(self.children)
        # print('backprop matmul')
        # print(gradient.shape)
        # print([feature.shape for feature in features])
        # if variable in self.children:
        #     print('kron substitution')
        #     print(np.swapaxes(gradient, -1, -2).shape)
        #     print([feature.shape for i, feature in
        #            enumerate(features) if variable is not self.children[i]])
        # else:
        #     print('standard backprop')
        #     print(gradient.shape)
        #     print([np.swapaxes(feature, -1, -2).shape for i, feature in
        #            enumerate(features) if variable is not self.children[i]])

        # apply identity to bypass kronecker
        if variable is self.children[0]:
            return np.swapaxes(gradient, -1, -2) @ np.swapaxes(features[1], -1, -2)

        # TODO: s @ W not tested
        if variable is self.children[1]:
            return np.swapaxes(gradient, -1, -2) @ features[0]

        # Take derivative of left side
        if variable in self.children[0].variables:
            return gradient @ np.swapaxes(features[1], -1, -2)

        # Take derivative of right side
        if variable in self.children[1].variables:
            return gradient @ features[0]

    def __str__(self):
        return ' @ '.join([str(child) for child in self.children])


class Transpose(Node):
    @property
    def output_nodes(self):
        raise NotImplementedError

    def propagate(self, features):
        return np.swapaxes(features, -1, -2)

    def backpropagate(self, features, variable, gradient):
        return np.swapaxes(gradient, -1, -2)

    def __str__(self):
        return f'({str(self.children[0])}).T'
