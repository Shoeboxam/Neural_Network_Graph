from Neural_Network import *
from Environments.base import Environment


class Dataset(Environment):

    def __init__(self, **dataframes):
        lengths = set(len(dataframes[tag]) for tag in dataframes)
        if len(lengths) != 1:
            raise ValueError('dataframes do not have equal size')

        self._number_rows = next(iter(lengths))
        self._dataframes = dataframes

        self.viewpoint = np.random.randint(0, 360)

    def sample(self, quantity=None):
        quantity = quantity or 1
        indices = np.random.randint(0, self._number_rows, size=quantity)
        return {tag: self._dataframes[tag][indices][..., None] for tag in self._dataframes}

    def survey(self, quantity=None):
        quantity = min(quantity or 256, self._number_rows)
        indices = np.linspace(0, self._number_rows - 1, quantity, dtype=np.uint)
        return {tag: self._dataframes[tag][indices][..., None] for tag in self._dataframes}

    def output_nodes(self, tag):
        return self._dataframes[tag].shape[1]

    def __len__(self):
        return self._number_rows

    @staticmethod
    def error(expect, predict):
        return np.linalg.norm(expect - predict)
