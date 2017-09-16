import numpy as np
from .Operator import Add, Matmul


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
        if type(children) is not list:
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
