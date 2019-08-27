from Environments.base import PlotError
from Environments.Figlet_Fonts import FigletFonts

from Neural_Network import *
import Neural_Network.optimizer as optimizers
from Neural_Network.optimizer_private import make_private_optimizer

from Tests.utils import pytest_utility


def test_figlet_autoencoder(plot=False):

    ascii_vals = [i for i in range(33, 126)]
    font = 'banner3'
    environment = FigletFonts(font, noise=0.0, autoencoder=True, ascii_vals=ascii_vals)

    # ~~~ Create the network ~~~
    domain = Source(environment, 'stimulus')
    codomain = Source(environment, 'expected')

    # Layer one
    weight_1 = Variable(np.random.normal(size=(100, domain.output_nodes)))
    biases_1 = Variable(np.random.normal(size=(100, 1)))
    hidden_1 = Bent(weight_1 @ domain + biases_1)

    # Layer two
    weight_2 = Variable(np.random.normal(size=(environment.output_nodes(), hidden_1.output_nodes)))
    biases_2 = Variable(np.random.normal(size=(environment.output_nodes(), 1)))
    graph = Logistic(weight_2 @ hidden_1 + biases_2)

    # Loss
    loss = SumSquared(graph, codomain)

    print("Network Summary:")
    print(str(graph))

    optimizer_class = optimizers.Adagrad
    # optimizer_class = make_private_optimizer(
    #     optimizer_class,
    #     epsilon=1, delta=1e-5,
    #     clipping_interval=10,
    #     num_rows=1000000)

    optimizer = optimizer_class(loss, rate=.01)

    plotters = [
        PlotError(111, environment.error)
    ] if plot else None

    error = pytest_utility(environment, optimizer, graph, plotters, iterations=None)
    print('Error:', error)
