from Neural_Network.Gate import *
from Neural_Network.Variable import Variable


# I broke the Transform into elementary operations-- now this needs rewriting
class TransformRecurrent(Transform):
    def __init__(self, children, nodes, decay=1.0):
        super().__init__(children, nodes)
        self.decay = decay
        self._variables['internal'] = Variable(np.random.normal(size=(nodes, nodes)))

        self._weight_gradient = np.zeros((self.output_nodes, self.input_nodes))
        self._bias_gradient = np.zeros((self.output_nodes, 1))
        self._internal_gradient = np.zeros((self.output_nodes, 1))

        self._prediction = np.zeros((self.output_nodes, 1))

    @cache
    def propagate(self, features):
        recurrent = self._variables['internal'] @ self._prediction
        features = np.vstack((features, self.decay * recurrent))

        self._prediction = self._variables['weights'] @ features + self._variables['biases']
        return self._prediction

    def backpropagate(self, features, variable, grad):
        if variable is self._variables['weights']:
            return grad.T @ features[None] @ (1 + self._variables['internal'])
        if variable is self._variables['biases']:
            return grad @ (1 + self._variables['internal'])
        if variable is self._variables['internal']:
            return grad.T @ features[None]
        # Gradients on recurrent nodes don't get passed back
        return grad[:, self.output_nodes:] @ self._variables['weights']

    @property
    def input_nodes(self):
        return super().input_nodes + self.output_nodes
