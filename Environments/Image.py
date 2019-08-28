import numpy as np
from Environments.base import Environment

from PIL import Image


class ImageSingle(Environment):
    def __init__(self, path):
        self.data = np.array(Image.open(path)).astype(float)

    def sample(self, quantity=None):
        return {'stimulus': self.data, 'expected': np.array(1)}

    def survey(self, quantity=None):
        return {'stimulus': self.data, 'expected': np.array(1)}

    def output_nodes(self, tag):
        if tag is 'stimulus':
            return self.data.shape
        return [1]

    def plot(self, plt, predict):
        plt.imshow(predict)
