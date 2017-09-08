import numpy as np


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

    def __call__(self, stimulus, parent=None):
        features = np.vstack([child(stimulus, self) for child in self.children])

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
        if variable not in self.variables:
            return grad @ np.zeros(self(stimulus).shape)

        # Gradients passed along a branch need to be sliced to the portion relevant to the child
        gradients = []
        cursor = 0
        for child in self.children:
            gradients.append(child.gradient(stimulus, variable, grad[:, cursor:cursor + child.output_nodes]))
            cursor += child.output_nodes
        return np.vstack(gradients)

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


class Stimulus(Gate):
    def __init__(self, environment):
        super().__init__(children=[])
        self.environment = environment

    @cache
    def __call__(self, stimulus):
        return stimulus[self.environment.tag]

    @property
    def gradient(self, stimulus, variable, grad):
        if variable is self.environment:
            return np.eye(self.output_nodes)
        return np.zeros([self.output_nodes] * 2)

    @property
    def output_nodes(self):
        return self.environment.size_output()
