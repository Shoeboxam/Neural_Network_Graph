# There is only one environment for the network.
# The environment may produce many sources when sampled.

import abc


class Environment(object):
    # return a random batch of samples from the environment (for training)
    @abc.abstractmethod
    def sample(self, quantity=None):
        pass

    # return a deterministic batch of samples from the environment (for plotting)
    @abc.abstractmethod
    def survey(self, quantity=None):
        pass

    @abc.abstractmethod
    def output_nodes(self, tag):
        pass

    def plot(self, plt, predict):
        pass
