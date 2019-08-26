import pandas

from Neural_Network import *
from Environments.Dataset import Dataset
import Neural_Network.optimizer as optimizers
from Neural_Network.optimizer_private import make_private_optimizer
from Tests.utils import pytest_utility


def test_pums(plot=False):
    dataframe = pandas.read_csv('/home/shoe/Desktop/MaPUMS5full.csv')

    environment = Dataset(
        stimulus=dataframe[['educ']].to_numpy(),
        expected=dataframe[['married']].to_numpy())

    # data sources
    stimulus = Source(environment, 'stimulus')
    expected = Source(environment, 'expected')
    # (only two classes for y, so encoding not necessary)

    # Layer one
    weight_1 = Variable(np.random.uniform(size=(1, stimulus.output_nodes)))
    biases_1 = Variable(np.random.uniform(size=(1, 1)))
    graph = Logistic(weight_1 @ stimulus + biases_1)

    # Loss
    loss = CrossEntropy(graph, expected)

    optimizer_class = optimizers.GradientDescent
    optimizer_class = make_private_optimizer(
        optimizer_class,
        epsilon=1, delta=1e-5,
        clipping_interval=10,
        num_rows=len(environment))

    optimizer = optimizer_class(loss, rate=.0001)

    error = pytest_utility(environment, optimizer, graph, plot)
    print('error', error)


def test_pums_multisource(plot=False):
    dataframe = pandas.read_csv('/home/shoe/Desktop/MaPUMS5full.csv')
    environment = Dataset(
        stimulus_1=dataframe[['educ']].to_numpy(),
        stimulus_2=dataframe[['sex', 'age', 'income', 'latino']].to_numpy(),
        expected=dataframe[['married']].to_numpy())

    # data sources
    stimulus_1 = Source(environment, 'stimulus_1')
    stimulus_2 = Source(environment, 'stimulus_2')
    expected = Source(environment, 'expected')

    # Layer one
    weight_1 = Variable(np.random.uniform(size=(4, stimulus_1.output_nodes)))
    biases_1 = Variable(np.random.uniform(size=(4, 1)))
    hidden_1 = Bent(weight_1 @ stimulus_1 + biases_1 + stimulus_2 * 2)

    # Layer two
    weight_2 = Variable(np.random.uniform(size=(2, stimulus_2.output_nodes)))
    biases_2 = Variable(np.random.uniform(size=(2, 1)))
    graph = Softmax(weight_2 @ hidden_1 + biases_2)

    # Loss
    loss = CrossEntropy(graph, expected)
    optimizer = optimizers.GradientDescent(loss)

    error = pytest_utility(environment, optimizer, graph, plot)
    print('error', error)


def test_boston(plot=False):
    try:
        from sklearn.datasets import load_boston
    except ImportError:
        print('Sklearn not installed. Install Sklearn to test with this dataset.')
        return

    loaded = load_boston()
    environment = Dataset(
        data=(loaded['data'] - np.mean(loaded['data'])) / np.std(loaded['data']),
        target=loaded['target'][:, None])

    # data sources
    data = Source(environment, 'data')
    target = Source(environment, 'target')

    # Layer one
    weight_1 = Variable(np.random.uniform(size=(100, data.output_nodes)))
    biases_1 = Variable(np.random.uniform(size=(100, 1)))
    hidden_1 = Softplus(weight_1 @ data + biases_1)

    # Layer two
    weight_2 = Variable(np.random.uniform(size=(1, hidden_1.output_nodes)))
    biases_2 = Variable(np.random.uniform(size=(1, 1)))
    graph = Softplus(weight_2 @ hidden_1 + biases_2)

    # Loss
    loss = SumSquared(graph, target)
    optimizer = optimizers.GradientDescent(loss, rate=.001)

    error = pytest_utility(environment, optimizer, graph, plot)

    assert error < 100
