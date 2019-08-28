import config
config.NP_BACKEND = 'JAX'

from Neural_Network import *
from Environments.Dataset import Dataset

import numpy
from jax import grad


def test_iris():
    try:
        from sklearn.datasets import load_iris
    except ImportError:
        print('Sklearn not installed. Install Sklearn to test with this dataset.')
        return

    loaded = load_iris()
    environment = Dataset(
        data=(loaded['data'] - np.mean(loaded['data'])) / np.std(loaded['data']),
        target=loaded['target'][:, None])

    # data sources
    data = Source(environment, 'data')
    target = Source(environment, 'target')

    raw_weights = np.array(numpy.random.uniform(size=(4, data.output_nodes)))
    network = None
    weight_1 = None
    survey = environment.survey()

    def net_with_parameter(weight_1_raw):
        nonlocal weight_1
        nonlocal network
        # Layer one
        weight_1 = Variable(weight_1_raw, label='weights_1')
        biases_1 = Variable(np.array(numpy.random.uniform(size=(4, 1))), label='biases_1')
        hidden_1 = Softplus(weight_1 @ data + biases_1)

        # Layer two
        weight_2 = Variable(np.array(numpy.random.uniform(size=(1, hidden_1.output_nodes))), label='weights_2')
        biases_2 = Variable(np.array(numpy.random.uniform(size=(1, 1))), label='biases_2')
        graph = Softplus(weight_2 @ hidden_1 + biases_2)

        network = SumSquared(graph, target)
        return network

    def jax_wrap(weight_1_raw):
        network_temp = net_with_parameter(weight_1_raw)
        return np.sum(network_temp(survey))

    gradient_jax = grad(jax_wrap)(raw_weights)
    gradient_custom = network.gradient(survey, weight_1)

    print('jax autograd')
    print(gradient_jax)
    print('custom grad')
    print(gradient_custom)

    assert np.allclose(gradient_jax, gradient_custom)
