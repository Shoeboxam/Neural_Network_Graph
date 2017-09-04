import numpy as np


# Return cached value if already computed for current stimulus
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


class Gate(object):
    def __init__(self, children):
        if type(children) is not list:
            children = [children]
        self.children = children

        # Stores references to variables in the gate
        self._variables = {}

        self._stored_variables = None
        self._stored_input_nodes = None
        self._stored_output_nodes = None

        self._cached_stimulus = {}
        self._cached___call__ = None
        self._cached_gradient = {}

    def __call__(self, stimulus):
        return np.vstack([child(stimulus) for child in self.children])

    def gradient(self, stimulus, variable, grad):
        if variable not in self.variables:
            return grad @ np.zeros(self(stimulus).shape)

        gradients = []
        cursor = 0
        for child in self.children:
            gradients.append(child.gradient(stimulus, variable, grad[:, cursor:cursor + child.output_nodes]))
            cursor += child.output_nodes
        return np.vstack(gradients)

    @property
    @_store
    def output_nodes(self):
        return sum([child.output_nodes for child in self.children])

    @property
    @_store
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
    @_store
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

    @_cache
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
