from Neural_Network import *
from Environments.Figlet_Fonts import FigletFonts

from Tests.utils import train_utility


def test_figlet_autoencoder(queue=None):

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

    error = train_utility(environment, loss, graph, queue=queue, iterations=3000)
    print('Error:', error)
