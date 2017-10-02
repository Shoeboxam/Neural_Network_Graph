from ..Gate import *
import numpy as np


class Source(Gate):
    def __init__(self):
        super().__init__(children=[])

    @cache
    def __call__(self, stimulus, parent=None):
        print(self.__class__.__name__)
        print(stimulus[self.tag_stimulus])
        return stimulus[self.tag_stimulus]

    @property
    def backpropagate(self, features, variable, grad):
        if variable is self:
            return np.eye(self.output_nodes)
        return np.zeros([self.output_nodes] * 2)

    @property
    def output_nodes(self):
        raise NotImplementedError("Source is an abstract base class.")

    def sample(self, quantity):
        raise NotImplementedError("Source is an abstract base class.")

    def survey(self, quantity):
        raise NotImplementedError("Source is an abstract base class.")

    @property
    def input_nodes(self):
        # The source node takes two arguments:
        #   First:  0 - source; 1 - label
        #   Second: Input tag
        return 2

    @property
    def output_nodes(self):
        raise NotImplementedError("Source is an abstract base class.")

    @property
    def output_labels(self):
        raise NotImplementedError("Source is an abstract base class.")

    def plot(self, plt, predict):
        pass

    @staticmethod
    def error(expect, predict):
        return np.linalg.norm(expect - predict)
