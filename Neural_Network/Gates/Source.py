from ..Gate import *
import numpy as np

# This is one source for the network.
# Each source instance handles a single tag from the environment class.


class Source(Gate):
    def __init__(self, source, tag):
        super().__init__(children=[])
        self.source = source
        self.tag = tag

    # Source is the network seed, so branching and pass-through logic is unnecessary and overriden
    def __call__(self, stimulus, parent=None):
        return self.propagate(stimulus)

    def propagate(self, stimulus):
        return stimulus[self.tag]

    @property
    def backpropagate(self, features, variable, grad):
        if variable is self:
            return np.eye(self.output_nodes)
        return np.zeros([self.output_nodes] * 2)

    @property
    def output_nodes(self):
        return self.source.output_nodes(tag=self.tag)

    @property
    def input_nodes(self):
        # The source node takes two arguments:
        #   First:  0 - source; 1 - label
        #   Second: Input tag
        return 2

    def __str__(self):
        return self.tag
