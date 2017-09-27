import numpy as np

_scalar = [str, int, float]


# Return cached value if already computed for current stimulus
def cache(method):
    def decorator(self, *args, **kwargs):
        if self._cached_stimulus == args[0]:
            self._cached_stimulus = args[0]
            return lambda **kw: getattr(self, '_cached_' + method.__name__)
        feature = method(self, *args, **kwargs)

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


# Slice the gradient to just the portion relative to a given branch
def grad_slice(method):
    def decorator(self, features, variable, gradient):
        gradient = method(self, features, variable, gradient)

        derivatives = {}
        cursor = 0

        for child in self.children:
            if variable in child.variables:
                derivatives[child] = gradient[:, cursor:cursor + child.output_nodes]
            cursor += child.output_nodes
        return derivatives
    return decorator


class Gate(object):
    def __init__(self, children):
        if type(children) not in [list, tuple]:
            children = [children]
        self.children = children

        self.parents = []
        for child in children:
            child.parents.append(self)

        self._stored_variables = None
        self._stored_input_nodes = None
        self._stored_output_nodes = None

        self._cached_stimulus = {}
        self._cached___call__ = None
        self._cached_gradient = {}

    # Forward pass
    def __call__(self, stimulus, parent=None):
        features = self.propagate([child(stimulus, self) for child in self.children])

        # Split a feature vector with respect to multiple parents
        if parent in self.parents:
            cursor = 0

            for par in self.parents:
                if parent is par:
                    return features[cursor:cursor + parent.input_nodes]
                cursor += par.input_nodes
        return features

    def gradient(self, stimulus, variable, grad):
        derivatives = self.backpropagate([child(stimulus, self) for child in self.children], variable, grad)

        # Gradients passed along a branch need to be sliced to the portion relevant to the child
        accumulator = []

        # Either the derivative is complete, or compute the complete derivative
        for child in self.children:
            if child in derivatives.keys():
                if variable is child:
                    accumulator.append(derivatives[variable])
                elif variable in child.variables:
                    accumulator.append(child.gradient(stimulus, variable, derivatives[child]))

        # Sum derivative for all instances of variable in the branch
        return np.sum(accumulator)

    # Define propagation in child classes
    def propagate(self, features):
        raise NotImplementedError("Gate is an abstract base class, and propagate is not defined.")

    def backpropagate(self, features, variable, gradient):
        raise NotImplementedError("Gate is an abstract base class, and backpropagate is not defined.")

    @property
    @store
    def output_nodes(self):
        return sum([child.output_nodes for child in self.children])

    @property
    @store
    def input_nodes(self):
        """Count the number of input nodes"""
        node_count = 0
        for child in self.children:
            if child.__class__.__name__ in ['Transform', 'Stimulus']:
                node_count += child.output_nodes
            else:
                node_count += child.input_nodes
        return node_count

    @property
    @store
    def variables(self):
        """List the input variables"""
        variables = []
        [variables.extend(child.variables) for child in self.children]
        return variables

    def __matmul__(self, other):
        return Matmul((self, other))

    def __add__(self, other):
        return Add((self, other))


class Add(Gate):
    def propagate(self, features):
        left = features[0]
        right = features[1]

        if type(left) in _scalar or type(right) in _scalar:
            return left + right

        # Implicitly cast lesser operand to a higher conformable dimension
        # Stimuli become vectorized, but bias units remain 1D. To add wx + b, must broadcast
        elif left.ndim == 2 and right.ndim == 1:
            return np.add(left, np.tile(right[..., np.newaxis], left.shape[1]))
        elif left.ndim == 1 and right.ndim == 2:
            return np.add(np.tile(left[..., np.newaxis], right.shape[1]), right)

        elif left.ndim == 3 and right.ndim == 2:
            return np.add(left, np.tile(right[..., np.newaxis], left.shape[2]))
        elif left.ndim == 2 and right.ndim == 3:
            return np.add(np.tile(left[..., np.newaxis], right.shape[2]), right)
        else:
            return left + right

    def backpropagate(self, features, variable, gradient):
        derivatives = {}

        for child in self.children:
            if variable is child or variable in child.variables:
                derivatives[child] = gradient
        return derivatives


class Matmul(Gate):
    @property
    def output_nodes(self):
        return self.children[0].output_nodes

    def propagate(self, features):
        left = features[0]
        right = features[0]

        if type(left) in _scalar or type(right) in _scalar:
            return left * right

        # Implicitly broadcast and vectorize matrix multiplication along axis 3
        # Matrix multiplication between 3D arrays is the matrix multiplication between respective matrix slices
        if left.ndim > 2 or right.ndim > 2:
            return np.einsum('ij...,jl...->il...', left, right)
        else:
            return left @ right

    def backpropagate(self, features, variable, gradient):
        derivatives = {}

        if variable is self.children[0]:
            derivatives[self.children[0]] = gradient.T @ features[1][None]
        elif variable in self.children[0].variables:
            derivatives[self.children[0]] = gradient @ features[1]

        if variable is self.children[1]:
            derivatives[self.children[1]] = gradient.T @ features[0][None]
        elif variable in self.children[1].variables:
            derivatives[self.children[1]] = gradient @ features[0]

        return derivatives
