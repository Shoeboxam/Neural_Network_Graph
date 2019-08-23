# Learn a continuous function
from Environments.Environment import Environment

from inspect import signature

import numpy as np


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

    def sample(self, quantity=None):
        quantity = quantity or 1

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

        # move batch to the first index, add trailing singleton axis to make column vector
        stimulus = np.moveaxis(stimulus, -1, 0)[:, :, None]
        expectation = np.moveaxis(np.array(expectation), -1, 0)[:, :, None]

        return {'stimulus': stimulus, 'expected': expectation}

    def survey(self, quantity=None):
        quantity = quantity or 128

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

        # move batch to the first index, add trailing singleton axis to make column vector
        stimulus = np.moveaxis(stimulus, -1, 0)[:, :, None]
        expectation = np.moveaxis(np.array(expectation), -1, 0)[:, :, None]

        return {'stimulus': stimulus, 'expected': expectation}

    def output_nodes(self, tag):
        if tag is 'stimulus':
            return self._size_stimulus

        if tag is 'expected':
            return self._size_expected

    def plot(self, plt, predict):
        survey = self.survey()
        x = survey['stimulus']
        y = survey['expected']

        print(x.shape)
        print(y.shape)

        # Output of function is 1 dimensional
        if y.shape[1] == 1:
            ax = plt.subplot(1, 2, 2)
            plt.ylim(self._range[0])

            ax.plot(x[:, 0], y[:, 0], marker='.', color=(0.3559, 0.7196, 0.8637))
            ax.plot(x[:, 0], predict[:, 0], marker='.', color=(.9148, .604, .0945))

        # Output of function has arbitrary dimensions
        if y.shape[1] > 1:

            ax = plt.subplot(1, 2, 2, projection='3d')
            plt.title('Environment')
            ax.scatter(x[:, 0], y[:, 0], y[:, 1], color=(0.3559, 0.7196, 0.8637))
            ax.scatter(x[:, 0], predict[:, 0], predict[:, 1], color=(.9148, .604, .0945))
            ax.view_init(elev=10., azim=self.viewpoint)
            self.viewpoint += 5

    @staticmethod
    def error(expect, predict):
        return np.linalg.norm(expect - predict)
