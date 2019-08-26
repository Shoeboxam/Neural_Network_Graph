from Neural_Network import *
import Neural_Network.optimizer as optimizers
from Environments.Continuous_Function import ContinuousLine, ContinuousScatter
from Neural_Network.optimizer_private import make_private_optimizer

from Tests.utils import pytest_utility


def test_continuous_3d_elbow(plot=False):

    environment = ContinuousScatter([
        lambda a, b: (2 * b**2 + 0.5 * a**3),
        lambda a, b: (0.5 * a**3 + 2 * b**2 - b)
    ], domain=[[-1, 1]] * 2)

    # environment = Continuous([lambda a: (24 * a**2 + a),
    #                           lambda a: (-5 * a**3)], domain=[[-1, 1]])

    # environment = Continuous([lambda x: np.cos(1 / x)], domain=[[-1, 1]])

    # environment = Continuous([lambda v: (24 * v**4 - 2 * v**2 + v)], domain=[[-1, 1]])

    # ~~~ Create the network ~~~

    domain = Source(environment, 'stimulus')
    codomain = Source(environment, 'expected')

    # Layer one
    weight_1 = Variable(np.random.uniform(size=(4, domain.output_nodes)))
    biases_1 = Variable(np.random.uniform(size=(4, 1)))
    hidden_1 = Bent(weight_1 @ domain + biases_1)

    # Layer two
    weight_2 = Variable(np.random.uniform(size=(2, hidden_1.output_nodes)))
    biases_2 = Variable(np.random.uniform(size=(2, 1)))
    graph = Bent(weight_2 @ hidden_1 + biases_2)

    # Loss
    loss = SumSquared(graph, codomain)

    print("Network Summary:")
    print(str(graph))

    optimizer_class = optimizers.GradientDescent
    optimizer_class = make_private_optimizer(
        optimizer_class,
        epsilon=1, delta=1e-5,
        clipping_interval=10,
        num_rows=1000)

    optimizer = optimizer_class(loss, rate=.01)

    error = pytest_utility(environment, optimizer, graph, plot, iterations=3000)
    print('Error:', error)
    assert error < 2


def test_continuous_sideways_saddle(plot=False):
    environment = ContinuousScatter([
        lambda a, b: (24 * a**2 - 2 * b**2 + a),
        lambda a, b: (12 * a ** 2 + 12 * b ** 3 + b)
    ], domain=[[-1, 1]] * 2)

    # ~~~ Create the network ~~~

    domain = Source(environment, 'stimulus')
    codomain = Source(environment, 'expected')

    # Layer one
    weight_1 = Variable(np.random.uniform(size=(100, domain.output_nodes)), label='weight_1')
    biases_1 = Variable(np.random.uniform(size=(100, 1)), label='bias_1')
    hidden_1 = Softplus(weight_1 @ domain + biases_1)

    # Layer two
    weight_2 = Variable(np.random.uniform(size=(2, hidden_1.output_nodes)), label='weight_2')
    biases_2 = Variable(np.random.uniform(size=(2, 1)), label='bias_2')
    graph = Softplus(weight_2 @ hidden_1 + biases_2)

    # Loss
    loss = SumSquared(graph, codomain)

    print("Network Summary:")
    print(str(graph))

    optimizer_class = optimizers.GradientDescent
    optimizer_class = make_private_optimizer(
        optimizer_class,
        epsilon=1, delta=1e-5,
        clipping_interval=10,
        num_rows=1000)

    optimizer = optimizer_class(loss, rate=.01)

    error = pytest_utility(environment, optimizer, graph, plot, iterations=3000)


def test_continuous_periodic(plot=False):

    environment = ContinuousLine([
        lambda x: np.sin(x),
        lambda x: np.cos(x)
    ], domain=[
        [-2 * np.pi, 2 * np.pi],
        [-np.pi, np.pi]
    ])

    # ~~~ Create the network ~~~

    domain = Source(environment, 'stimulus')
    codomain = Source(environment, 'expected')

    # Layer one
    weight_1 = Variable(np.random.uniform(size=(4, domain.output_nodes)))
    biases_1 = Variable(np.random.uniform(size=(4, 1)))
    hidden_1 = Sinusoidal(weight_1 @ domain + biases_1)

    # Layer two
    weight_2 = Variable(np.random.uniform(size=(2, hidden_1.output_nodes)))
    biases_2 = Variable(np.random.uniform(size=(2, 1)))
    graph = Sinusoidal(weight_2 @ hidden_1 + biases_2)

    # Loss
    loss = SumSquared(graph, codomain)

    print("Network Summary:")
    print(str(graph))

    optimizer_class = optimizers.GradientDescent
    optimizer_class = make_private_optimizer(
        optimizer_class,
        epsilon=1, delta=1e-5,
        clipping_interval=10,
        num_rows=1000)

    optimizer = optimizer_class(loss, rate=.001)

    error = pytest_utility(environment, optimizer, graph, plot, iterations=5000)
    assert error < 1
