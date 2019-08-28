from config import NP_BACKEND
from Neural_Network.node import Node

print('loading backend', NP_BACKEND)
if NP_BACKEND == 'NUMPY':
    import numpy as np
elif NP_BACKEND == 'JAX':
    import jax.numpy as np

# This is one source for the network.
# Each source instance handles a single tag from the environment class.


class Source(Node):
    def __init__(self, source, tag):
        super().__init__(children=[])
        self.source = source
        self.tag = tag

    # Source is a leaf in the network, so branching and pass-through logic is unnecessary and overriden
    def __call__(self, stimulus, parent=None):
        propagated = self.propagate(stimulus)
        if NP_BACKEND == 'JAX':
            return np.array(propagated[0])
        return propagated

    def propagate(self, stimulus):
        return stimulus[self.tag]

    def backpropagate(self, features, variable, grad):
        if variable is self:
            return np.eye(self.output_nodes)
        return np.zeros([self.output_nodes] * 2)

    @property
    def output_nodes(self):
        return self.source.output_nodes(tag=self.tag)

    def __str__(self):
        return self.tag
