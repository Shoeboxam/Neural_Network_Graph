# Learn a continuous function
from inspect import signature
from Neural_Network import *

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

plt.style.use('fivethirtyeight')
plot_points = []
np.set_printoptions(suppress=True, linewidth=10000)


class Continuous(Environment):

    def __init__(self, funct, domain, range=None):

        self._size_stimulus = len(signature(funct[0]).parameters)
        self._size_expected = len(funct)

        self._funct = funct
        self._domain = domain

        if range is None:
            self._range = [[-1, 1]] * len(funct)

            if self._size_stimulus == 1 and self._size_expected == 1:
                candidates = self._funct[0](np.linspace(*self._domain[0], num=100))
                self._range = [[min(candidates), max(candidates)]]
        else:
            self._range = range

        self.viewpoint = np.random.randint(0, 360)

    @Environment._tag
    def sample(self, quantity):
        if quantity is None:
            quantity = 1

        # Generate random values for each input stimulus
        axes = []
        for idx in range(self._size_stimulus):
            axes.append(np.random.uniform(low=self._domain[idx][0], high=self._domain[idx][1], size=quantity))

        meshgrid = np.array(np.meshgrid(*axes)).reshape(self._size_stimulus, -1)

        # Subsample meshgrid
        axes_selections = np.random.randint(meshgrid.shape[1], size=quantity)
        stimulus = meshgrid[:, axes_selections]

        # Evaluate each function with the stimuli
        expectation = []
        for idx, f in enumerate(self._funct):
            expectation.append(f(*stimulus))

        return [np.array(stimulus), np.array(expectation)]

    @Environment._tag
    def survey(self, quantity):
        if quantity is None:
            quantity = 128

        # Quantity is adjusted to the closest meshgrid approximation
        axis_length = int(round(quantity ** self._size_stimulus ** -1))

        axes = []
        for idx in range(self._size_stimulus):
            axes.append(np.linspace(start=self._domain[idx][0], stop=self._domain[idx][1], num=axis_length))

        stimulus = np.array(np.meshgrid(*axes)).reshape(self._size_stimulus, -1)

        # Evaluate each function with the stimuli
        expectation = []
        for idx, f in enumerate(self._funct):
            expectation.append(f(*stimulus))

        return [np.array(stimulus), np.array(expectation)]

    def output_nodes(self, tag):
        if tag is 'stimulus':
            return self._size_stimulus

        if tag is 'expected':
            return self._size_expected

    def plot(self, plt, predict):
        survey = self.survey()
        x = survey['stimulus']
        y = survey['expected']

        # Output of function is 1 dimensional
        if y.shape[0] == 1:
            ax = plt.subplot(1, 2, 2)
            plt.ylim(self._range[0])

            ax.plot(x[0], y[0], marker='.', color=(0.3559, 0.7196, 0.8637))
            ax.plot(x[0], predict[0], marker='.', color=(.9148, .604, .0945))

        # Output of function has arbitrary dimensions
        if y.shape[0] > 1:

            ax = plt.subplot(1, 2, 2, projection='3d')
            plt.title('Environment')
            ax.scatter(x[0], y[0], y[1], color=(0.3559, 0.7196, 0.8637))
            ax.scatter(x[0], predict[0], predict[1], color=(.9148, .604, .0945))
            ax.view_init(elev=10., azim=self.viewpoint)
            self.viewpoint += 5

    @staticmethod
    def error(expect, predict):
        return np.linalg.norm(expect - predict)

# environment = Continuous([lambda a, b: (24 * a**2 - 2 * b**2 + a),
#                           lambda a, b: (12 * a ** 2 + 12 * b ** 3 + b)], domain=[[-1, 1]] * 2)

environment = Continuous([lambda a, b: (2 * b**2 + 0.5 * a**3),
                          lambda a, b: (0.5 * a**3 + 2 * b**2 - b)], domain=[[-1, 1]] * 2)

# environment = Continuous([lambda x: np.sin(x),
#                           lambda x: np.cos(x)], domain=[[-2 * np.pi, 2 * np.pi], [-np.pi, np.pi]])

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
variables = graph.variables

print("Network Summary:")
print(str(graph))
# ~~~ Train the network ~~~
step = .01
i = 0

while True:
    i += 1
    sample = environment.sample(quantity=20)

    for variable in graph:
        variable -= step * np.average(loss.gradient(sample, variable), axis=2)

    if i % 50 == 0:
        survey = environment.survey()
        prediction = graph(survey)

        error = environment.error(survey['expected'], prediction)
        plot_points.append((i, error))

        # Error plot
        plt.subplot(1, 2, 1)
        plt.cla()
        plt.title('Error')
        plt.plot(*zip(*plot_points), marker='.', color=(.9148, .604, .0945))

        # Environment plot
        plt.subplot(1, 2, 2)
        plt.cla()
        plt.title('Environment')
        environment.plot(plt, prediction)

        plt.pause(0.00001)
