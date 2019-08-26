# Learn a continuous function
from Environments.Environment import Environment, ScatterMixin, PlotMixin

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
        stimulus = np.atleast_3d(np.moveaxis(stimulus, -1, 0))
        expectation = np.atleast_3d(np.moveaxis(np.array(expectation), -1, 0))

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
        stimulus = np.atleast_3d(np.moveaxis(stimulus, -1, 0))
        expectation = np.atleast_3d(np.moveaxis(np.array(expectation), -1, 0))

        return {'stimulus': stimulus, 'expected': expectation}

    def output_nodes(self, tag):
        if tag is 'stimulus':
            return self._size_stimulus

        if tag is 'expected':
            return self._size_expected

    @staticmethod
    def error(expect, predict):
        return np.linalg.norm(expect - predict)


class ContinuousLine(PlotMixin, Continuous):
    pass


class ContinuousScatter(ScatterMixin, Continuous):
    pass
