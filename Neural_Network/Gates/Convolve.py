from ..Gate import *

import numpy as np


class Convolve(Gate):
    def __init__(self, children, kernel):
        super().__init__(children)
        self.kernel = kernel

    @cache
    def propagate(self, features):
        # Combine features from all children
        features = np.vstack(features)

        # Determine edge of kernel
        kernel_edge = np.array(self.kernel.shape) - 1
        if len(features.shape) == 3:
            kernel_edge = np.append(kernel_edge, np.array([0]))

        # Convert 3D image into 5D sampler [kernel elements, channel, sample index]
        shape = [*self.kernel.shape, *(features.shape - kernel_edge)]
        memory_offset = [*features.strides[:2], *features.strides]
        samples = np.lib.stride_tricks.as_strided(features, shape=shape, strides=memory_offset)

        # Contract over first two dimensions
        return np.einsum("ij...,ij->...", samples, self.kernel)

    @grad_slice
    @cache
    def backpropagate(self, features, variable, grad):
        # Not implemented
        return grad
