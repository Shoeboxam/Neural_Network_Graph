import numpy as np
from .Operators import Add, Matmul


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


class Gate(object):
    def __init__(self, children):
        if type(children) is not list:
            children = [children]
        self.children = children

        self.parents = []
        for child in children:
            child.parents.append(self)

        # Stores references to variables in the gate
        self._variables = {}

        self._stored_variables = None
        self._stored_input_nodes = None
        self._stored_output_nodes = None

        self._cached_stimulus = {}
        self._cached___call__ = None
        self._cached_gradient = {}

    # Forward pass
    def __call__(self, stimulus, parent=None):
        features = np.vstack([child(stimulus, self) for child in self.children])

        features = self.propagate(features)

        # Split a feature vector with respect to multiple parents
        if parent in self.parents:
            cursor = 0

            for par in self.parents:
                if parent is par:
                    break
                cursor += par.input_nodes
            features = features[cursor:cursor + parent.input_nodes, :]
        return features

    def gradient(self, stimulus, variable, grad):
        features = np.vstack([child(stimulus, self) for child in self.children])

        grad = self.backpropagate(features, variable, grad)

        # Variables cannot be reused in the same branch of the same network
        if variable in self._variables:
            return grad

        # Gradients passed along a branch need to be sliced to the portion relevant to the child
        gradients = []
        cursor = 0
        for child in self.children:
            if variable in child.variables:
                gradients.append(child.gradient(stimulus, variable, grad[:, cursor:cursor + child.output_nodes]))
                cursor += child.output_nodes
            else:
                gradients.append(np.zeros())
        return np.vstack(gradients)

    # Define propagation in child classes
    def propagate(self, features):
        return features

    def backpropagate(self, features, variable, gradient):
        return gradient

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
        variables = list(self._variables.values())
        for child in self.children:
            variables.extend(child.variables)
        return variables

    def __matmul__(self, other):
        return Matmul((self, other))

    def __add__(self, other):
        return Add((self, other))


class Source(Gate):
    def __init__(self, environment):
        super().__init__(children=[])
        self.environment = environment

    @cache
    def propagate(self, stimulus):
        return stimulus[self.environment.tag]

    @property
    def backpropagate(self, features, variable, grad):
        if variable is self.environment:
            return np.eye(self.output_nodes)
        return np.zeros([self.output_nodes] * 2)

    @property
    def output_nodes(self):
        return self.environment.size_output()
