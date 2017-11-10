# There is only one environment for the network.
# The environment may produce many sources when sampled.


class Environment(object):

    def _tag(method):
        def decorator(_self, quantity=None, tagged=True):
            if tagged:
                return dict(zip(_self.tags, method(_self, quantity)))
            return method(_self, quantity)
        return decorator

    # These are the labels for each of the indices returned by the sample and survey methods
    tags = ['stimulus', 'expected']

    @_tag
    def sample(self, quantity):
        raise NotImplementedError("Environment is abstract.")

    @_tag
    def survey(self, quantity):
        raise NotImplementedError("Environment is abstract.")

    def output_nodes(self, tag):
        raise NotImplementedError("Environment is abstract")

    def plot(self, plt, predict):
        pass
