from ..Gate import *


class Stack(Gate):
    def propagate(self, features):
        return np.vstack(features)

    def gradient(self, stimulus, variable, grad):
        for child in self.children:
            if variable is child:
                return np.eye(child.output_nodes)
            if variable in child.children:
                return child.gradient(stimulus, variable, grad)


class Split(Gate):

    def __init__(self, children, node_interval):
        super().__init__(children)
        self.node_interval = node_interval

    def propagate(self, features):
        # Split a feature vector with respect to multiple parents
        return np.vstack(features)[self.node_interval[0]:self.node_interval[1]]

    def gradient(self, stimulus, variable, grad):
        gradient = super().gradient(stimulus, variable, grad)
        return gradient[self.node_interval[0]:self.node_interval[1]]
