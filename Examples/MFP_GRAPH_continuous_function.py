# Learn a continuous function
from inspect import signature
from Neural_Network import *

import numpy as np
np.set_printoptions(suppress=True, linewidth=10000)


class Continuous(Source):

    def __init__(self, funct, domain, range=None):
        super().__init__()

        self._size_input = len(signature(funct[0]).parameters)
        self._size_output = len(funct)

        self._funct = funct
        self._domain = domain

        self.tag_expected = 'Continuous_expected_' + str(np.random.randint(1000, 9999))
        self.tag_stimulus = 'Continuous_stimulus_' + str(np.random.randint(1000, 9999))

        if range is None:
            self._range = [[-1, 1]] * len(funct)

            if self._size_input == 1 and self._size_output == 1:
                candidates = self._funct[0](np.linspace(*self._domain[0], num=100))
                self._range = [[min(candidates), max(candidates)]]
        else:
            self._range = range

        self.viewpoint = np.random.randint(0, 360)

    def sample(self, quantity=1):
        # Generate random values for each input stimulus

        axes = []
        for idx in range(self._size_input):
            axes.append(np.random.uniform(low=self._domain[idx][0], high=self._domain[idx][1], size=quantity))

        meshgrid = np.array(np.meshgrid(*axes)).reshape(self._size_input, -1)

        # Subsample meshgrid
        axes_selections = np.random.randint(meshgrid.shape[1], size=quantity)
        stimulus = meshgrid[:, axes_selections]

        # Evaluate each function with the stimuli
        expectation = []
        for idx, f in enumerate(self._funct):
            expectation.append(f(*stimulus))

        return [np.array(stimulus), np.array(expectation)]

    def survey(self, quantity=128):
        # Quantity is adjusted to the closest meshgrid approximation
        axis_length = int(round(quantity**self._size_input**-1))

        axes = []
        for idx in range(self._size_input):
            axes.append(np.linspace(start=self._domain[idx][0], stop=self._domain[idx][1], num=axis_length))

        stimulus = np.array(np.meshgrid(*axes)).reshape(self._size_input, -1)

        # Evaluate each function with the stimuli
        expectation = []
        for idx, f in enumerate(self._funct):
            expectation.append(f(*stimulus))

        return [np.array(stimulus), np.array(expectation)]

    @property
    def size_input(self):
        return self._size_input

    @property
    def output_nodes(self):
        return self._size_output

    def plot(self, plt, predict):
        x, y = self.survey()

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

# environment = Continuous([lambda a, b: (24 * a**2 - 2 * b**2 + a),
#                           lambda a, b: (12 * a ** 2 + 12 * b ** 3 + b)], domain=[[-1, 1]] * 2)

environment = Continuous([lambda a, b: (2 * b**2 + 0.5 * a**3 + 50),
                          lambda a, b: (0.5 * a**3 + 2 * b**2 - b - 23)], domain=[[-1, 1]] * 2)

# environment = Continuous([lambda x: np.sin(x),
#                           lambda x: np.cos(x)], domain=[[-2 * np.pi, 2 * np.pi], [-np.pi, np.pi]])

# environment = Continuous([lambda a: (24 * a**2 + a),
#                           lambda a: (-5 * a**3)], domain=[[-1, 1]])

# environment = Continuous([lambda x: np.cos(1 / x)], domain=[[-1, 1]])

# environment = Continuous([lambda v: (24 * v**4 - 2 * v**2 + v)], domain=[[-1, 1]])

# ~~~ Create the network ~~~
weight = Variable(np.random.uniform(size=(4, environment.output_nodes)))
bias = Variable(np.random.uniform(size=(4, 1)))

transform = weight @ environment + bias
graph = Logistic(transform)
loss = SumSquared(graph, environment)
variables = graph.variables

# ~~~ Test the network ~~~
[stimuli, expectation] = environment.sample()
print(loss.gradient({environment.tag_stimulus: stimuli}, {environment.tag_expected: expectation}, weight))

print(graph.gradient({environment.tag_stimulus: stimuli}, weight, np.ones([1, 2])))
print(graph({environment.tag_stimulus: stimuli}))
